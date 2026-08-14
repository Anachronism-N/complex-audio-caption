"""Import locally decoded MUSDB18-HQ stems with per-track license metadata."""

from __future__ import annotations

import csv
from pathlib import Path

from sceneledger.data.source_catalog import SourceRecord

MUSDB18_DATASET = "MUSDB18-HQ"
OFFICIAL_TRACKLIST_URL = (
    "https://raw.githubusercontent.com/sigsep/website/master/"
    "content/datasets/assets/tracklist.csv"
)
DEFAULT_LICENSE_MAP = {
    # Preserve the official table literally when it does not state a
    # Creative Commons version.  Inferring 4.0 here would manufacture legal
    # metadata that the published tracklist does not contain.
    "CC BY-NC-SA": "CC BY-NC-SA",
    "CC BY-NC-SA 3.0": "CC BY-NC-SA 3.0",
    "Restricted": "MUSDB18 educational use only",
}
VALIDATION_TRACKS = {
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
}


def read_musdb18_tracklist(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"Track Name", "Genre", "Source", "License"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"invalid MUSDB18 tracklist: {path}")
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row["Track Name"]).strip()
        if not name or name in output:
            raise ValueError(f"duplicate/empty MUSDB18 track name: {name!r}")
        output[name] = {key: str(value).strip() for key, value in row.items()}
    return output


def _split(subset: str, track_name: str) -> str:
    normalized = subset.casefold()
    if normalized == "test":
        return "test"
    if normalized == "train":
        return "val" if track_name in VALIDATION_TRACKS else "train"
    raise ValueError(f"unsupported MUSDB18 subset directory: {subset}")


def convert_musdb18_records(
    root: str | Path,
    *,
    tracklist_path: str | Path,
    allowed_licenses: set[str],
) -> list[SourceRecord]:
    """Create one accompaniment and one vocal source per licensed song."""
    if not allowed_licenses:
        raise ValueError("MUSDB18 track licenses must be explicitly allowlisted")
    base = Path(root).expanduser().resolve()
    tracklist = read_musdb18_tracklist(tracklist_path)
    records: list[SourceRecord] = []
    observed_tracks: set[str] = set()
    for subset in ("train", "test"):
        subset_root = base / subset
        if not subset_root.is_dir():
            raise FileNotFoundError(f"MUSDB18 subset is missing: {subset_root}")
        for track_root in sorted(path for path in subset_root.iterdir() if path.is_dir()):
            name = track_root.name
            metadata = tracklist.get(name)
            if metadata is None:
                raise ValueError(f"MUSDB18 track missing official license metadata: {name}")
            observed_tracks.add(name)
            license_name = DEFAULT_LICENSE_MAP.get(metadata["License"], metadata["License"])
            if license_name not in allowed_licenses:
                continue
            accompaniment = track_root / "accompaniment.wav"
            vocals = track_root / "vocals.wav"
            for path in (accompaniment, vocals):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"decoded MUSDB18 stem is missing for {name}: {path.name}"
                    )
            group = f"musdb18-track:{name.casefold()}"
            split = _split(subset, name)
            genre = metadata["Genre"] or "unknown genre"
            common = {
                "source_group": group,
                "leakage_groups": [f"musdb18-artist:{name.partition(' - ')[0].casefold()}"],
                "dataset": MUSDB18_DATASET,
                "license": license_name,
                "annotation_origin": "dataset",
                "attribution": f"MUSDB18 track: {name}; source={metadata['Source']}",
                "original_url": "https://sigsep.github.io/datasets/musdb.html",
                "split": split,
            }
            records.extend(
                [
                    SourceRecord(
                        source_id=f"musdb18:{subset}:{name}:accompaniment",
                        kind="music",
                        audio_path=accompaniment.relative_to(base).as_posix(),
                        labels=[genre, "instrumental_accompaniment"],
                        caption=f"Instrumental {genre} accompaniment without lead vocals.",
                        **common,
                    ),
                    SourceRecord(
                        source_id=f"musdb18:{subset}:{name}:vocals",
                        kind="vocal",
                        audio_path=vocals.relative_to(base).as_posix(),
                        labels=[genre, "isolated_vocals"],
                        caption=f"Isolated vocals from a {genre} song; lyrics are not transcribed.",
                        **common,
                    ),
                ]
            )
    if not observed_tracks:
        raise ValueError("MUSDB18 conversion discovered zero tracks")
    if not records:
        raise ValueError("MUSDB18 conversion selected zero licensed tracks")
    return records


__all__ = [
    "DEFAULT_LICENSE_MAP",
    "MUSDB18_DATASET",
    "OFFICIAL_TRACKLIST_URL",
    "VALIDATION_TRACKS",
    "convert_musdb18_records",
    "read_musdb18_tracklist",
]
