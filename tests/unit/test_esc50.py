from __future__ import annotations

import csv
from pathlib import Path

from sceneledger.data.esc50 import (
    ESC10_LICENSE,
    ESC50_NONCOMMERCIAL_LICENSE,
    convert_esc50_records,
    read_esc50_metadata,
)


def test_esc50_conversion_preserves_fold_group_label_and_license(tmp_path: Path) -> None:
    metadata = tmp_path / "esc50.csv"
    fields = ["filename", "fold", "target", "category", "esc10", "src_file", "take"]
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"filename": "1-100-a.wav", "fold": 1, "target": 0, "category": "dog", "esc10": True, "src_file": 100, "take": "a"},
                {"filename": "4-200-a.wav", "fold": 4, "target": 10, "category": "rain", "esc10": False, "src_file": 200, "take": "a"},
                {"filename": "5-300-a.wav", "fold": 5, "target": 11, "category": "chainsaw", "esc10": False, "src_file": 300, "take": "a"},
            ]
        )

    records = convert_esc50_records(read_esc50_metadata(metadata), require_complete=False)

    assert [record.split for record in records] == ["train", "val", "test"]
    assert records[0].labels == ["dog"]
    assert records[0].source_group == "freesound-clip:100"
    assert records[0].license == ESC10_LICENSE
    assert records[1].kind == "ambience"
    assert records[1].license == ESC50_NONCOMMERCIAL_LICENSE
    assert records[2].kind == "sfx"


def test_esc50_complete_mode_fails_on_partial_metadata() -> None:
    row = {
        "filename": "1-100-a.wav",
        "fold": 1,
        "target": 0,
        "category": "dog",
        "esc10": True,
        "src_file": 100,
        "take": "a",
    }
    try:
        convert_esc50_records([row])
    except ValueError as exc:
        assert "expected complete ESC-50 metadata" in str(exc)
    else:
        raise AssertionError("partial ESC-50 metadata was accepted")
