from __future__ import annotations

import json
from pathlib import Path

import pytest

from sceneledger.cli.catalog_librispeech import build_librispeech_catalog


def _make_utterance(root: Path, utterance_id: str, text: str) -> None:
    speaker, chapter, _ = utterance_id.split("-", maxsplit=2)
    directory = root / "LibriSpeech" / "dev-clean" / speaker / chapter
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{utterance_id}.flac").write_bytes(b"not decoded by the cataloger")
    transcript = directory / f"{speaker}-{chapter}.trans.txt"
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(f"{utterance_id} {text}\n")


def test_build_librispeech_catalog_groups_all_utterances_by_speaker(tmp_path: Path):
    _make_utterance(tmp_path, "19-198-0000", "HELLO WORLD")
    _make_utterance(tmp_path, "19-198-0001", "A SECOND UTTERANCE")
    output = tmp_path / "catalog.jsonl"

    report = build_librispeech_catalog(
        root=tmp_path,
        subsets=["dev-clean"],
        output=output,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["n_sources"] == 2
    assert report["n_source_groups"] == 1
    assert {row["source_group"] for row in rows} == {"librispeech-speaker-19"}
    assert {row["text"] for row in rows} == {"HELLO WORLD", "A SECOND UTTERANCE"}
    assert all(row["kind"] == "speech" and row["verbatim"] is True for row in rows)
    assert all(row["license"] == "CC BY 4.0" for row in rows)


def test_build_librispeech_catalog_rejects_missing_waveform(tmp_path: Path):
    directory = tmp_path / "LibriSpeech" / "dev-clean" / "19" / "198"
    directory.mkdir(parents=True)
    (directory / "19-198.trans.txt").write_text(
        "19-198-0000 HELLO WORLD\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="missing waveform"):
        build_librispeech_catalog(
            root=tmp_path,
            subsets=["dev-clean"],
            output=tmp_path / "catalog.jsonl",
        )
