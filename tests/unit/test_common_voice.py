from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sceneledger.data.common_voice import (
    COMMON_VOICE_LICENSE,
    convert_common_voice_records,
)


def _write_split(root: Path, name: str, rows: list[dict[str, str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "clips").mkdir(exist_ok=True)
    fields = ["client_id", "path", "sentence", "up_votes", "down_votes", "locale"]
    with (root / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        (root / "clips" / row["path"]).write_bytes(b"placeholder")


def _row(speaker: str, path: str, sentence: str) -> dict[str, str]:
    return {
        "client_id": speaker,
        "path": path,
        "sentence": sentence,
        "up_votes": "2",
        "down_votes": "0",
        "locale": "zh-CN",
    }


def test_common_voice_import_preserves_exact_text_and_hides_client_id(tmp_path: Path) -> None:
    _write_split(tmp_path, "train.tsv", [_row("private-a", "a.mp3", "你好")])
    _write_split(tmp_path, "dev.tsv", [_row("private-b", "b.mp3", "早上好")])
    _write_split(tmp_path, "test.tsv", [_row("private-c", "c.mp3", "谢谢")])

    records = convert_common_voice_records(
        tmp_path, release="cv-corpus-22.0", locale="zh-CN"
    )

    assert {record.split for record in records} == {"train", "val", "test"}
    assert {record.caption for record in records} == {"你好", "早上好", "谢谢"}
    assert all(record.text_is_verbatim for record in records)
    assert all(record.license == COMMON_VOICE_LICENSE for record in records)
    assert all("private-" not in record.source_group for record in records)
    assert all(record.language == "zh-CN" for record in records)


def test_common_voice_import_rejects_speaker_overlap(tmp_path: Path) -> None:
    _write_split(tmp_path, "train.tsv", [_row("same-speaker", "a.mp3", "one")])
    _write_split(tmp_path, "dev.tsv", [_row("different", "b.mp3", "two")])
    _write_split(tmp_path, "test.tsv", [_row("same-speaker", "c.mp3", "three")])

    with pytest.raises(ValueError, match="speaker overlap"):
        convert_common_voice_records(
            tmp_path, release="cv-corpus-22.0", locale="zh-CN"
        )


def test_common_voice_import_rejects_path_escape(tmp_path: Path) -> None:
    row = _row("speaker-a", "../outside.mp3", "one")
    _write_split(tmp_path, "train.tsv", [row])
    _write_split(tmp_path, "dev.tsv", [_row("speaker-b", "b.mp3", "two")])
    _write_split(tmp_path, "test.tsv", [_row("speaker-c", "c.mp3", "three")])
    (tmp_path / "outside.mp3").write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="escapes clips"):
        convert_common_voice_records(
            tmp_path, release="cv-corpus-22.0", locale="zh-CN"
        )
