"""Import LibriSpeech utterances with transcripts and speaker leakage groups."""

from __future__ import annotations

from pathlib import Path

from sceneledger.data.source_catalog import SourceRecord

LIBRISPEECH_DATASET = "LibriSpeech"
LIBRISPEECH_LICENSE = "CC BY 4.0"

SUBSET_TO_SPLIT = {
    "train-clean-5": "train",
    "train-clean-100": "train",
    "train-clean-360": "train",
    "train-other-500": "train",
    "dev-clean-2": "val",
    "dev-clean": "val",
    "dev-other": "val",
    "test-clean": "test",
    "test-other": "test",
}


def _discover_subsets(root: Path, requested: set[str] | None) -> list[Path]:
    candidates = sorted(
        directory
        for directory in root.rglob("*")
        if directory.is_dir() and directory.name in SUBSET_TO_SPLIT
    )
    if root.name in SUBSET_TO_SPLIT:
        candidates.insert(0, root)
    unique = {path.resolve(): path.resolve() for path in candidates}
    selected = [path for path in unique.values() if requested is None or path.name in requested]
    observed = {path.name for path in selected}
    missing = sorted((requested or set()) - observed)
    if missing:
        raise FileNotFoundError(f"requested LibriSpeech subsets were not found: {missing}")
    if not selected:
        raise FileNotFoundError(f"no supported LibriSpeech subsets found under {root}")
    return sorted(selected)


def _read_transcripts(subset: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for transcript_file in sorted(subset.rglob("*.trans.txt")):
        for line_no, line in enumerate(transcript_file.read_text(encoding="utf-8").splitlines(), 1):
            utterance_id, separator, text = line.strip().partition(" ")
            if not separator or not text.strip():
                raise ValueError(f"invalid transcript {transcript_file}:{line_no}")
            if utterance_id in transcripts:
                raise ValueError(f"duplicate LibriSpeech utterance transcript: {utterance_id}")
            transcripts[utterance_id] = text.strip()
    if not transcripts:
        raise ValueError(f"no transcript files found in {subset}")
    return transcripts


def convert_librispeech_records(
    root: str | Path,
    *,
    subsets: set[str] | None = None,
    max_per_speaker: int | None = None,
) -> list[SourceRecord]:
    """Create one exact-transcript source record per FLAC utterance."""
    if max_per_speaker is not None and max_per_speaker <= 0:
        raise ValueError("max_per_speaker must be positive")
    audio_root = Path(root).expanduser().resolve()
    subset_paths = _discover_subsets(audio_root, subsets)
    records: list[SourceRecord] = []
    speaker_counts: dict[tuple[str, str], int] = {}
    for subset in subset_paths:
        transcripts = _read_transcripts(subset)
        audio_by_id = {path.stem: path for path in sorted(subset.rglob("*.flac"))}
        missing_audio = sorted(set(transcripts) - set(audio_by_id))
        extra_audio = sorted(set(audio_by_id) - set(transcripts))
        if missing_audio or extra_audio:
            raise ValueError(
                f"LibriSpeech transcript/audio mismatch in {subset.name}: "
                f"missing_audio={missing_audio[:10]}, missing_transcript={extra_audio[:10]}"
            )
        for utterance_id, text in sorted(transcripts.items()):
            parts = utterance_id.split("-")
            if len(parts) != 3 or not all(part.isdigit() for part in parts):
                raise ValueError(f"unexpected LibriSpeech utterance ID: {utterance_id}")
            speaker_id, chapter_id, _utterance_index = parts
            count_key = (subset.name, speaker_id)
            if max_per_speaker is not None and speaker_counts.get(count_key, 0) >= max_per_speaker:
                continue
            speaker_counts[count_key] = speaker_counts.get(count_key, 0) + 1
            path = audio_by_id[utterance_id]
            records.append(
                SourceRecord(
                    source_id=f"librispeech:{subset.name}:{utterance_id}",
                    kind="speech",
                    audio_path=path.relative_to(audio_root).as_posix(),
                    source_group=f"librispeech-speaker:{speaker_id}",
                    leakage_groups=[
                        f"librispeech-book-chapter:{speaker_id}:{chapter_id}",
                    ],
                    labels=["read_english_speech"],
                    caption=text,
                    dataset=(
                        f"Mini {LIBRISPEECH_DATASET}/{subset.name}"
                        if subset.name in {"train-clean-5", "dev-clean-2"}
                        else f"{LIBRISPEECH_DATASET}/{subset.name}"
                    ),
                    license=LIBRISPEECH_LICENSE,
                    annotation_origin="dataset",
                    text_is_verbatim=True,
                    identity=f"speaker:{speaker_id}",
                    language="en",
                    attribution="LibriSpeech, derived from LibriVox audiobooks",
                    original_url=(
                        "https://www.openslr.org/31/"
                        if subset.name in {"train-clean-5", "dev-clean-2"}
                        else "https://www.openslr.org/12/"
                    ),
                    split=SUBSET_TO_SPLIT[subset.name],
                )
            )
    if not records:
        raise ValueError("LibriSpeech conversion selected zero utterances")
    return records


__all__ = [
    "LIBRISPEECH_DATASET",
    "LIBRISPEECH_LICENSE",
    "SUBSET_TO_SPLIT",
    "convert_librispeech_records",
]
