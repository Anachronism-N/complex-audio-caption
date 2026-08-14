from __future__ import annotations

from pathlib import Path

from sceneledger.data.librispeech import LIBRISPEECH_LICENSE, convert_librispeech_records


def _write_subset(root: Path, subset: str, speaker: str, chapter: str) -> None:
    directory = root / "LibriSpeech" / subset / speaker / chapter
    directory.mkdir(parents=True)
    utterance_id = f"{speaker}-{chapter}-0001"
    (directory / f"{utterance_id}.trans.txt").write_text(
        f"{utterance_id} THE EXACT TRANSCRIPT\n", encoding="utf-8"
    )
    (directory / f"{utterance_id}.flac").write_bytes(b"placeholder")


def test_librispeech_import_preserves_transcript_speaker_and_split(tmp_path: Path) -> None:
    _write_subset(tmp_path, "train-clean-5", "19", "198")
    _write_subset(tmp_path, "dev-clean-2", "20", "199")
    _write_subset(tmp_path, "test-clean", "21", "200")

    records = convert_librispeech_records(tmp_path)

    assert [record.split for record in records] == ["val", "test", "train"]
    train = next(record for record in records if record.split == "train")
    assert train.caption == "THE EXACT TRANSCRIPT"
    assert train.identity == "speaker:19"
    assert train.source_group == "librispeech-speaker:19"
    assert train.leakage_groups == ["librispeech-book-chapter:19:198"]
    assert train.license == LIBRISPEECH_LICENSE
    assert train.annotation_origin == "dataset"
    assert train.text_is_verbatim is True
    assert train.audio_path.endswith("19-198-0001.flac")


def test_librispeech_import_rejects_transcript_audio_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "LibriSpeech" / "test-clean" / "21" / "200"
    directory.mkdir(parents=True)
    (directory / "21-200.trans.txt").write_text(
        "21-200-0001 MISSING AUDIO\n", encoding="utf-8"
    )
    try:
        convert_librispeech_records(tmp_path)
    except ValueError as exc:
        assert "transcript/audio mismatch" in str(exc)
    else:
        raise AssertionError("missing LibriSpeech audio was accepted")
