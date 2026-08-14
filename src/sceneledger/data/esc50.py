"""Convert official ESC-50 metadata into an auditable source catalog.

ESC-50 is an isolated-event sanity anchor, not a complex-scene benchmark.  We
preserve its official folds and Freesound source identifiers so downstream
experiments cannot silently leak related excerpts across splits.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sceneledger.data.source_catalog import SourceRecord

ESC50_DATASET = "ESC-50"
ESC50_NONCOMMERCIAL_LICENSE = "CC BY-NC 3.0"
ESC10_LICENSE = "CC BY 3.0"
ESC50_FOLD_TO_SPLIT = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}

# A project-level operational mapping.  Sounds with a sustained environmental
# bed are ambience; discrete objects, actions, animals and human non-speech are
# SFX.  The original 50-way class label is always retained in ``labels``.
ESC50_AMBIENCE_CATEGORIES = {
    "chirping_birds",
    "crackling_fire",
    "crickets",
    "pouring_water",
    "rain",
    "sea_waves",
    "thunderstorm",
    "water_drops",
    "wind",
}

REQUIRED_COLUMNS = {
    "filename",
    "fold",
    "target",
    "category",
    "esc10",
    "src_file",
    "take",
}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def _find_metadata(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    csv_candidates = sorted(path.rglob("esc50.csv"))
    if csv_candidates:
        return [csv_candidates[0]]
    parquet_candidates = sorted(path.rglob("*.parquet"))
    if parquet_candidates:
        return parquet_candidates
    raise FileNotFoundError(f"no esc50.csv or parquet shards found under {path}")


def read_esc50_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Read the official CSV, or compatible Hugging Face parquet shards."""
    files = _find_metadata(Path(path).expanduser().resolve())
    if files[0].suffix.lower() == ".csv":
        with files[0].open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional server dependency
            raise RuntimeError(
                "parquet metadata requires pandas+pyarrow; alternatively use official meta/esc50.csv"
            ) from exc
        rows = []
        for file in files:
            rows.extend(pd.read_parquet(file).to_dict(orient="records"))
    if not rows:
        raise ValueError("ESC-50 metadata is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"ESC-50 metadata is missing columns: {sorted(missing)}")
    filenames = [str(row["filename"]).strip() for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("ESC-50 metadata contains duplicate filenames")
    return rows


def convert_esc50_records(
    rows: list[dict[str, Any]],
    *,
    require_complete: bool = True,
) -> list[SourceRecord]:
    """Create label-only records while preserving official split provenance."""
    if require_complete:
        folds = {int(row["fold"]) for row in rows}
        categories = {str(row["category"]).strip() for row in rows}
        if len(rows) != 2000 or folds != {1, 2, 3, 4, 5} or len(categories) != 50:
            raise ValueError(
                "expected complete ESC-50 metadata (2000 clips, 50 categories, folds 1..5); "
                f"observed clips={len(rows)}, categories={len(categories)}, folds={sorted(folds)}"
            )

    records: list[SourceRecord] = []
    for row in rows:
        filename = str(row["filename"]).strip()
        category = str(row["category"]).strip()
        fold = int(row["fold"])
        if fold not in ESC50_FOLD_TO_SPLIT:
            raise ValueError(f"unsupported ESC-50 fold {fold} for {filename}")
        source_file = str(row["src_file"]).strip()
        human_label = category.replace("_", " ")
        records.append(
            SourceRecord(
                source_id=f"esc50:{Path(filename).stem}",
                kind="ambience" if category in ESC50_AMBIENCE_CATEGORIES else "sfx",
                audio_path=filename,
                source_group=f"freesound-clip:{source_file}",
                labels=[category],
                caption=f"ESC-50 class label: {human_label}.",
                dataset=ESC50_DATASET,
                license=ESC10_LICENSE if _bool(row["esc10"]) else ESC50_NONCOMMERCIAL_LICENSE,
                annotation_origin="dataset",
                attribution=f"ESC-50 excerpt from Freesound source {source_file}",
                original_url=f"https://freesound.org/s/{source_file}/",
                split=ESC50_FOLD_TO_SPLIT[fold],
            )
        )
    return records


__all__ = [
    "ESC10_LICENSE",
    "ESC50_AMBIENCE_CATEGORIES",
    "ESC50_DATASET",
    "ESC50_FOLD_TO_SPLIT",
    "ESC50_NONCOMMERCIAL_LICENSE",
    "convert_esc50_records",
    "read_esc50_metadata",
]
