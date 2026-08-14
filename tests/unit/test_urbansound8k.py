from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sceneledger.data.urbansound8k import (
    STRICT_FOREGROUND_CLASSES,
    URBANSOUND8K_LICENSE,
    convert_urbansound8k_records,
)

FIELDS = [
    "slice_file_name",
    "fsID",
    "start",
    "end",
    "salience",
    "fold",
    "classID",
    "class",
]


def _write_dataset(root: Path, rows: list[dict[str, str]]) -> None:
    metadata = root / "UrbanSound8K" / "metadata"
    metadata.mkdir(parents=True)
    with (metadata / "UrbanSound8K.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        audio = root / "UrbanSound8K" / "audio" / f"fold{row['fold']}"
        audio.mkdir(parents=True, exist_ok=True)
        (audio / row["slice_file_name"]).write_bytes(b"placeholder")


def _row(name: str, fsid: str, fold: int, category: str, salience: int = 1) -> dict[str, str]:
    return {
        "slice_file_name": name,
        "fsID": fsid,
        "start": "0.0",
        "end": "1.0",
        "salience": str(salience),
        "fold": str(fold),
        "classID": "1",
        "class": category,
    }


def test_urbansound8k_strict_import_keeps_only_foreground_discrete_events(
    tmp_path: Path,
) -> None:
    rows = [
        _row("a.wav", "100", 1, "car_horn"),
        _row("b.wav", "101", 7, "dog_bark"),
        _row("c.wav", "102", 9, "siren"),
        _row("d.wav", "103", 1, "street_music"),
        _row("e.wav", "104", 1, "gun_shot", salience=2),
    ]
    _write_dataset(tmp_path, rows)

    audio_root, records = convert_urbansound8k_records(
        tmp_path, require_complete=False
    )

    assert audio_root.name == "UrbanSound8K"
    assert {record.labels[0] for record in records} == {"car_horn", "dog_bark", "siren"}
    assert {record.split for record in records} == {"train", "val", "test"}
    assert all(record.kind == "sfx" for record in records)
    assert all(record.license == URBANSOUND8K_LICENSE for record in records)
    assert all(record.source_group.startswith("freesound-clip:") for record in records)
    assert "street_music" not in STRICT_FOREGROUND_CLASSES


def test_urbansound8k_rejects_freesound_recording_across_splits(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        [
            _row("a.wav", "same", 1, "car_horn"),
            _row("b.wav", "same", 9, "siren"),
        ],
    )

    with pytest.raises(ValueError, match="crosses train/val/test"):
        convert_urbansound8k_records(tmp_path, require_complete=False)
