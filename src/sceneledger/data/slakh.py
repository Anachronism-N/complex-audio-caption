"""Import leakage-safe Slakh2100-redux instrumental mixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sceneledger.data.source_catalog import SourceRecord

SLAKH_LICENSE = "CC BY 4.0"
SLAKH_URL = "https://zenodo.org/records/4599666"
VALID_VARIANTS = {"Slakh2100-redux", "Slakh2100-split2"}
SPLIT_DIRS = {"train": "train", "validation": "val", "val": "val", "test": "test"}
VOICE_MARKERS = ("voice", "vocal", "choir", "aahs", "oohs")


def _metadata(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid Slakh metadata: {path}")
    return payload


def _instrument_labels(payload: dict[str, Any], path: Path) -> tuple[list[str], bool]:
    stems = payload.get("stems")
    if not isinstance(stems, dict) or not stems:
        raise ValueError(f"Slakh metadata has no stems: {path}")
    classes: set[str] = set()
    voice_like = False
    rendered = 0
    for stem_name, value in stems.items():
        if not isinstance(value, dict):
            raise ValueError(f"invalid Slakh stem {stem_name!r} in {path}")
        if not bool(value.get("audio_rendered")):
            continue
        rendered += 1
        fields = " ".join(
            str(value.get(key, ""))
            for key in ("inst_class", "midi_program_name", "plugin_name")
        ).casefold()
        voice_like = voice_like or any(marker in fields for marker in VOICE_MARKERS)
        label = str(value.get("inst_class", "")).strip()
        if label:
            classes.add(label)
    if rendered == 0 or not classes:
        raise ValueError(f"Slakh track has no rendered instrument metadata: {path}")
    return sorted(classes), voice_like


def convert_slakh_records(
    root: str | Path,
    *,
    split_variant: str,
    reject_voice_like: bool = True,
    min_instrument_classes: int = 4,
) -> list[SourceRecord]:
    """Create one semantic music track per Slakh instrumental mixture.

    The original Slakh split is intentionally rejected because its duplicate
    MIDI files cross train/evaluation boundaries.  UUID is retained as a
    leakage group even for the corrected split.
    """
    if split_variant not in VALID_VARIANTS:
        raise ValueError(
            f"split_variant must be one of {sorted(VALID_VARIANTS)}; "
            "the leakage-prone Slakh2100-orig split is forbidden"
        )
    if min_instrument_classes <= 0:
        raise ValueError("min_instrument_classes must be positive")
    base = Path(root).expanduser().resolve()
    records: list[SourceRecord] = []
    observed_splits: set[str] = set()
    observed_tracks: set[str] = set()
    uuid_splits: dict[str, set[str]] = {}

    for directory_name, split in SPLIT_DIRS.items():
        split_root = base / directory_name
        if not split_root.is_dir():
            continue
        if split == "val" and "val" in observed_splits:
            raise ValueError("Slakh root contains both val/ and validation/")
        observed_splits.add(split)
        for track_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
            if not track_root.name.startswith("Track"):
                continue
            if track_root.name in observed_tracks:
                raise ValueError(f"duplicate Slakh track directory: {track_root.name}")
            observed_tracks.add(track_root.name)
            metadata_path = track_root / "metadata.yaml"
            mix_path = track_root / "mix.flac"
            if not metadata_path.is_file() or not mix_path.is_file():
                raise FileNotFoundError(
                    f"incomplete Slakh track {track_root}: metadata.yaml/mix.flac required"
                )
            payload = _metadata(metadata_path)
            uuid = str(payload.get("UUID", "")).strip()
            if not uuid:
                raise ValueError(f"Slakh metadata has no UUID: {metadata_path}")
            uuid_splits.setdefault(uuid, set()).add(split)
            instruments, voice_like = _instrument_labels(payload, metadata_path)
            if voice_like and reject_voice_like:
                continue
            if len(instruments) < min_instrument_classes:
                continue
            readable = ", ".join(instruments[:-1])
            readable += f", and {instruments[-1]}" if len(instruments) > 1 else instruments[0]
            records.append(
                SourceRecord(
                    source_id=(
                        f"slakh:{split_variant.casefold()}:{split}:{track_root.name.casefold()}"
                    ),
                    kind="music",
                    audio_path=mix_path.relative_to(base).as_posix(),
                    source_group=f"slakh-track:{track_root.name.casefold()}",
                    leakage_groups=[f"slakh-midi:{uuid.casefold()}"],
                    labels=["synthetic_instrumental_music", *instruments],
                    caption=f"Synthetic instrumental music featuring {readable}.",
                    dataset=split_variant,
                    license=SLAKH_LICENSE,
                    annotation_origin="dataset",
                    text_is_verbatim=False,
                    attribution="Slakh2100; MERL and Northwestern University",
                    original_url=SLAKH_URL,
                    split=split,
                )
            )

    missing_splits = {"train", "val", "test"} - observed_splits
    if missing_splits:
        raise FileNotFoundError(f"Slakh corrected split directories missing: {sorted(missing_splits)}")
    cross_split = {
        uuid: sorted(splits) for uuid, splits in uuid_splits.items() if len(splits) > 1
    }
    if cross_split:
        raise ValueError(
            "Slakh MIDI UUID leakage remains across corrected splits: "
            f"{list(sorted(cross_split.items()))[:10]}"
        )
    if not records:
        raise ValueError("Slakh conversion selected zero instrumental tracks")
    return records


__all__ = [
    "SLAKH_LICENSE",
    "SLAKH_URL",
    "VALID_VARIANTS",
    "convert_slakh_records",
]
