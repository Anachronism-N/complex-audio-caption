"""Convert FSD50K weak labels and clip metadata into source records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sceneledger.data.source_catalog import SourceRecord

FSD50K_DATASET = "FSD50K"
FSD50K_DATASET_LICENSE = "CC BY 4.0"
LICENSE_MAP = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC BY 3.0",
    "https://creativecommons.org/licenses/by/3.0/": "CC BY 3.0",
    "http://creativecommons.org/licenses/by-nc/3.0/": "CC BY-NC 3.0",
    "https://creativecommons.org/licenses/by-nc/3.0/": "CC BY-NC 3.0",
    "http://creativecommons.org/licenses/sampling+/1.0/": "CC Sampling+ 1.0",
    "https://creativecommons.org/licenses/sampling+/1.0/": "CC Sampling+ 1.0",
}
MUSIC_LABELS = {
    "Music",
    "Musical_instrument",
    "Singing",
    "Choir",
    "Rapping",
}
SPEECH_LABELS = {
    "Speech",
    "Human_voice",
    "Whispering",
    "Speech_synthesizer",
}
AMBIENCE_LABELS = {
    "Rain",
    "Thunderstorm",
    "Water",
    "Waterfall",
    "Stream",
    "Ocean",
    "Waves_and_surf",
    "Wind",
    "Rustling_leaves",
    "Crackle",
    "Fire",
    "Environmental_noise",
    "Traffic_noise_and_roadway_noise",
    "Inside_small_room",
    "Outside_urban_or_manmade",
    "Outside_rural_or_natural",
}


def _read_ground_truth(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"fname", "labels", "mids"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"invalid FSD50K ground truth: {path}")
    for row in rows:
        row["resolved_split"] = row.get("split") or split
    return rows


def _read_metadata(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid FSD50K metadata: {path}")
    return {str(key): dict(value) for key, value in payload.items()}


def _read_ratings(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid FSD50K PP/PNP ratings: {path}")
    return {
        str(clip_id): {
            str(mid): [float(value) for value in values]
            for mid, values in dict(mid_ratings).items()
        }
        for clip_id, mid_ratings in payload.items()
    }


def _leaf_labels(labels: list[str]) -> list[str]:
    broad = {
        "Music",
        "Musical_instrument",
        "Human_sounds",
        "Human_voice",
        "Sounds_of_things",
        "Animal",
        "Natural_sounds",
        "Source-ambiguous_sounds",
    }
    leaves = [label for label in labels if label not in broad]
    return leaves or labels


def _kind(labels: set[str]) -> str | None:
    # FSD50K contains many music and speech clips, but without transcripts or
    # isolated stems they must not silently enter those source roles.
    if labels & (MUSIC_LABELS | SPEECH_LABELS):
        return None
    return "ambience" if labels & AMBIENCE_LABELS else "sfx"


def convert_fsd50k_records(
    root: str | Path,
    *,
    allowed_licenses: set[str],
    include_eval: bool = True,
    max_labels: int = 4,
) -> list[SourceRecord]:
    """Convert eligible non-speech/non-music clips and preserve uploader groups."""
    if not allowed_licenses:
        raise ValueError("FSD50K clip licenses must be explicitly allowlisted")
    base = Path(root).expanduser().resolve()
    ground_truth = base / "FSD50K.ground_truth"
    metadata_root = base / "FSD50K.metadata"
    rows = _read_ground_truth(ground_truth / "dev.csv", "train")
    metadata = _read_metadata(metadata_root / "dev_clips_info_FSD50K.json")
    if include_eval:
        rows.extend(_read_ground_truth(ground_truth / "eval.csv", "test"))
        metadata.update(_read_metadata(metadata_root / "eval_clips_info_FSD50K.json"))
    ratings = _read_ratings(metadata_root / "pp_pnp_ratings_FSD50K.json")

    eligible: list[tuple[dict[str, str], dict, str, list[str], str]] = []
    for row in rows:
        clip_id = str(row["fname"])
        clip = metadata.get(clip_id)
        if clip is None:
            raise ValueError(f"FSD50K metadata missing clip {clip_id}")
        raw_license = str(clip.get("license", ""))
        license_name = LICENSE_MAP.get(raw_license)
        if license_name is None:
            raise ValueError(f"unknown FSD50K clip license for {clip_id}: {raw_license!r}")
        if license_name not in allowed_licenses:
            continue
        labels = [item.strip() for item in str(row["labels"]).split(",") if item.strip()]
        mids = [item.strip() for item in str(row["mids"]).split(",") if item.strip()]
        if len(labels) != len(mids):
            raise ValueError(f"FSD50K label/MID mismatch for {clip_id}")
        kind = _kind(set(labels))
        if kind is None:
            continue
        label_by_mid = {mid: labels[index] for index, mid in enumerate(mids)}
        predominant = [
            label_by_mid[mid]
            for mid, values in ratings.get(clip_id, {}).items()
            if mid in label_by_mid and values.count(1.0) >= 2
        ]
        # PP twice is the documented high-agreement "present and predominant"
        # signal.  Requiring exactly one such label yields source-like clips;
        # broad/smeared labels remain provenance but not caption supervision.
        if len(predominant) != 1:
            continue
        uploader = str(clip.get("uploader", "")).strip()
        if not uploader:
            raise ValueError(f"FSD50K metadata missing uploader for {clip_id}")
        row = {**row, "predominant_label": predominant[0]}
        eligible.append((row, clip, license_name, labels, uploader))

    # The official dev train/val split contains uploader overlap.  Keep the
    # exhaustive eval set frozen as test, and move every conflicting dev
    # uploader wholly to train.  This preserves all eligible audio while
    # enforcing the stronger group-disjoint contract required by our mixer.
    splits_by_uploader: dict[str, set[str]] = {}
    for row, _clip, _license, _labels, uploader in eligible:
        splits_by_uploader.setdefault(uploader.casefold(), set()).add(
            str(row["resolved_split"])
        )
    if any("test" in splits and len(splits) > 1 for splits in splits_by_uploader.values()):
        raise ValueError("FSD50K eval uploader overlaps the development set")

    records: list[SourceRecord] = []
    for row, _clip, license_name, labels, uploader in eligible:
        clip_id = str(row["fname"])
        kind = _kind(set(labels))
        assert kind is not None
        predominant = str(row["predominant_label"])
        selected_labels = [
            predominant,
            *(label for label in _leaf_labels(labels) if label != predominant),
        ][:max_labels]
        human_label = predominant.replace("_", " ")
        split = str(row["resolved_split"])
        if splits_by_uploader[uploader.casefold()] == {"train", "val"}:
            split = "train"
        audio_folder = "FSD50K.eval_audio" if split == "test" else "FSD50K.dev_audio"
        records.append(
            SourceRecord(
                source_id=f"fsd50k:{clip_id}",
                kind=kind,  # type: ignore[arg-type]
                audio_path=f"{audio_folder}/{clip_id}.wav",
                source_group=f"freesound-uploader:{uploader.casefold()}",
                leakage_groups=[f"freesound-clip:{clip_id}"],
                labels=selected_labels,
                caption=f"FSD50K predominant class label: {human_label}.",
                dataset=FSD50K_DATASET,
                license=license_name,
                annotation_origin="dataset",
                attribution=f"Freesound clip {clip_id}, uploaded by {uploader}",
                original_url=f"https://freesound.org/s/{clip_id}/",
                split=split,  # type: ignore[arg-type]
            )
        )
    if not records:
        raise ValueError("FSD50K conversion selected zero eligible clips")
    group_splits: dict[str, set[str | None]] = {}
    for record in records:
        group_splits.setdefault(record.source_group, set()).add(record.split)
    conflicts = {
        group: splits for group, splits in group_splits.items() if len(splits) > 1
    }
    if conflicts:
        raise AssertionError(f"FSD50K uploader split conflicts remain: {conflicts}")
    return records


__all__ = [
    "FSD50K_DATASET",
    "FSD50K_DATASET_LICENSE",
    "LICENSE_MAP",
    "convert_fsd50k_records",
]
