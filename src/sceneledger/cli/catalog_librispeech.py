"""Convert an extracted LibriSpeech subset into a SceneLedger source catalog."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sceneledger.data.source_catalog import SourceRecord, write_source_catalog


def _corpus_root(root: str | Path) -> Path:
    candidate = Path(root).resolve()
    nested = candidate / "LibriSpeech"
    if nested.is_dir():
        return nested
    return candidate


def build_librispeech_catalog(
    *,
    root: str | Path,
    subsets: list[str],
    output: str | Path,
) -> dict:
    """Build a speaker-grouped catalog from official ``*.trans.txt`` files."""
    corpus_root = _corpus_root(root)
    if not subsets:
        raise ValueError("at least one LibriSpeech subset is required")

    records: list[SourceRecord] = []
    seen_utterances: set[str] = set()
    subset_counts: Counter[str] = Counter()
    speakers: set[str] = set()
    for subset in subsets:
        subset_root = corpus_root / subset
        if not subset_root.is_dir():
            raise FileNotFoundError(f"LibriSpeech subset not found: {subset_root}")
        transcript_files = sorted(subset_root.rglob("*.trans.txt"))
        if not transcript_files:
            raise FileNotFoundError(f"no LibriSpeech transcripts found under {subset_root}")
        for transcript_file in transcript_files:
            for line_number, raw_line in enumerate(
                transcript_file.read_text(encoding="utf-8").splitlines(), 1
            ):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split(maxsplit=1)
                if len(fields) != 2 or not fields[1].strip():
                    raise ValueError(
                        f"{transcript_file}:{line_number}: expected '<utterance-id> <text>'"
                    )
                utterance_id, transcript = fields[0], fields[1].strip()
                if utterance_id in seen_utterances:
                    raise ValueError(f"duplicate LibriSpeech utterance ID: {utterance_id}")
                speaker_id = utterance_id.split("-", maxsplit=1)[0]
                if not speaker_id:
                    raise ValueError(f"invalid LibriSpeech utterance ID: {utterance_id}")
                audio_path = transcript_file.parent / f"{utterance_id}.flac"
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"missing waveform for {utterance_id}: {audio_path}"
                    )
                seen_utterances.add(utterance_id)
                speakers.add(speaker_id)
                subset_counts[subset] += 1
                records.append(
                    SourceRecord(
                        path=str(audio_path.resolve()),
                        kind="speech",
                        text=transcript,
                        source_group=f"librispeech-speaker-{speaker_id}",
                        identity=f"librispeech-speaker-{speaker_id}",
                        language="en",
                        verbatim=True,
                        license="CC BY 4.0",
                        dataset=f"LibriSpeech/{subset}",
                    )
                )

    records.sort(key=lambda record: record.path)
    write_source_catalog(output, records)
    return {
        "corpus_root": str(corpus_root),
        "output": str(Path(output).resolve()),
        "subsets": list(subsets),
        "subset_counts": dict(sorted(subset_counts.items())),
        "n_sources": len(records),
        "n_source_groups": len(speakers),
        "grouping": "speaker",
        "license": "CC BY 4.0",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-catalog-librispeech")
    parser.add_argument(
        "--root",
        required=True,
        help="extracted directory containing LibriSpeech/ or the LibriSpeech directory itself",
    )
    parser.add_argument(
        "--subset",
        action="append",
        dest="subsets",
        required=True,
        help="subset to register; repeat this flag to combine multiple extracted subsets",
    )
    parser.add_argument("--output", required=True, help="output canonical JSONL catalog")
    parser.add_argument("--report", default=None, help="optional JSON audit report")
    args = parser.parse_args(argv)

    summary = build_librispeech_catalog(
        root=args.root,
        subsets=args.subsets,
        output=args.output,
    )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
