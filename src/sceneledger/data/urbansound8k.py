"""Import a strict foreground subset of the official UrbanSound8K release."""

from __future__ import annotations

import csv
from pathlib import Path

from sceneledger.data.source_catalog import SourceRecord

URBANSOUND8K_DATASET = "UrbanSound8K"
URBANSOUND8K_LICENSE = "CC BY-NC 3.0"
URBANSOUND8K_URL = "https://zenodo.org/records/1203745"
URBANSOUND8K_FOLD_TO_SPLIT = {
    1: "train",
    2: "train",
    3: "train",
    4: "train",
    5: "train",
    6: "train",
    7: "val",
    8: "val",
    9: "test",
    10: "test",
}

# These are foreground, source-like urban events.  Air conditioner and engine
# idling are sustained beds; children playing is multi-speaker; street music
# may contain vocals.  They are excluded rather than assigned unsupported dry
# source captions.
STRICT_FOREGROUND_CLASSES = {
    "car_horn",
    "dog_bark",
    "drilling",
    "gun_shot",
    "jackhammer",
    "siren",
}
REQUIRED_COLUMNS = {
    "slice_file_name",
    "fsID",
    "start",
    "end",
    "salience",
    "fold",
    "classID",
    "class",
}


def _metadata_path(root: Path) -> Path:
    candidates = sorted(root.rglob("UrbanSound8K.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one metadata/UrbanSound8K.csv under {root}, found {candidates}"
        )
    return candidates[0]


def read_urbansound8k_metadata(
    root: str | Path, *, require_complete: bool = True
) -> tuple[Path, list[dict[str, str]]]:
    base = Path(root).expanduser().resolve()
    metadata = _metadata_path(base)
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
            raise ValueError(
                f"invalid UrbanSound8K metadata columns: {reader.fieldnames}"
            )
        rows = [dict(row) for row in reader]
    filenames = [row["slice_file_name"].strip() for row in rows]
    if not rows or len(filenames) != len(set(filenames)):
        raise ValueError("UrbanSound8K metadata is empty or contains duplicate filenames")
    if require_complete:
        folds = {int(row["fold"]) for row in rows}
        classes = {row["class"].strip() for row in rows}
        if len(rows) != 8732 or folds != set(range(1, 11)) or len(classes) != 10:
            raise ValueError(
                "expected complete UrbanSound8K v1 metadata: "
                f"rows={len(rows)} folds={sorted(folds)} classes={len(classes)}"
            )
    dataset_root = metadata.parent.parent
    return dataset_root, rows


def convert_urbansound8k_records(
    root: str | Path,
    *,
    require_complete: bool = True,
    foreground_only: bool = True,
) -> tuple[Path, list[SourceRecord]]:
    """Create SFX records while preserving Freesound recording groups."""
    dataset_root, rows = read_urbansound8k_metadata(
        root, require_complete=require_complete
    )
    records: list[SourceRecord] = []
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        category = row["class"].strip()
        if category not in STRICT_FOREGROUND_CLASSES:
            continue
        salience = int(row["salience"])
        if foreground_only and salience != 1:
            continue
        fold = int(row["fold"])
        if fold not in URBANSOUND8K_FOLD_TO_SPLIT:
            raise ValueError(f"invalid UrbanSound8K fold: {fold}")
        filename = row["slice_file_name"].strip()
        audio = dataset_root / "audio" / f"fold{fold}" / filename
        if not audio.is_file():
            raise FileNotFoundError(f"UrbanSound8K audio is missing: {audio}")
        freesound_id = row["fsID"].strip()
        if not freesound_id:
            raise ValueError(f"UrbanSound8K row has no fsID: {filename}")
        split = URBANSOUND8K_FOLD_TO_SPLIT[fold]
        group = f"freesound-clip:{freesound_id}"
        group_splits.setdefault(group, set()).add(split)
        records.append(
            SourceRecord(
                source_id=f"urbansound8k:{Path(filename).stem}",
                kind="sfx",
                audio_path=audio.relative_to(dataset_root).as_posix(),
                source_group=group,
                labels=[category],
                caption=(
                    "UrbanSound8K foreground class label: "
                    f"{category.replace('_', ' ')}."
                ),
                dataset=URBANSOUND8K_DATASET,
                license=URBANSOUND8K_LICENSE,
                annotation_origin="dataset",
                text_is_verbatim=False,
                attribution=f"UrbanSound8K excerpt from Freesound source {freesound_id}",
                original_url=f"https://freesound.org/s/{freesound_id}/",
                split=split,  # type: ignore[arg-type]
            )
        )
    conflicts = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    if conflicts:
        raise ValueError(
            "UrbanSound8K Freesound recording crosses train/val/test: "
            f"{list(sorted(conflicts.items()))[:10]}"
        )
    if not records:
        raise ValueError("UrbanSound8K strict foreground conversion selected zero clips")
    return dataset_root, records


__all__ = [
    "STRICT_FOREGROUND_CLASSES",
    "URBANSOUND8K_DATASET",
    "URBANSOUND8K_FOLD_TO_SPLIT",
    "URBANSOUND8K_LICENSE",
    "URBANSOUND8K_URL",
    "convert_urbansound8k_records",
    "read_urbansound8k_metadata",
]
