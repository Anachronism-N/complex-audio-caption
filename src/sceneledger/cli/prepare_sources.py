"""Validate and canonicalize a real-source CSV/JSONL catalog."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sceneledger.data.manifests import file_hash
from sceneledger.data.source_catalog import load_source_catalog, write_source_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-prepare-sources")
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        dest="inputs",
        help="CSV or JSONL source metadata; repeat to merge multiple corpora",
    )
    parser.add_argument("--output", required=True, help="canonical JSONL catalog")
    parser.add_argument("--audio-root", default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="metadata-only audit; rendered experiments must not use this flag",
    )
    parser.add_argument(
        "--require-kind",
        action="append",
        default=[],
        choices=["speech", "vocal", "music", "sfx", "ambience"],
        help="fail unless this source kind is present; repeat for template coverage",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    records = []
    seen_paths: set[str] = set()
    for input_path in args.inputs:
        loaded = load_source_catalog(
            input_path,
            audio_root=args.audio_root,
            require_files=not args.allow_missing,
        )
        for record in loaded:
            if record.path in seen_paths:
                raise ValueError(
                    f"duplicate waveform across source catalogs: {record.path}"
                )
            seen_paths.add(record.path)
            records.append(record)
    kind_counts = Counter(record.kind for record in records)
    missing_kinds = sorted(kind for kind in args.require_kind if not kind_counts[kind])
    if missing_kinds:
        raise ValueError(f"source catalog is missing required kinds: {missing_kinds}")
    write_source_catalog(args.output, records)
    input_paths = [Path(path).resolve() for path in args.inputs]
    output_path = Path(args.output).resolve()
    summary = {
        "schema_version": "b3-source-catalog-v1",
        "inputs": [str(path) for path in input_paths],
        "input_sha256": {str(path): file_hash(path) for path in input_paths},
        "output": str(output_path),
        "output_sha256": file_hash(output_path),
        "audio_root": str(Path(args.audio_root).resolve()) if args.audio_root else None,
        "n_sources": len(records),
        "kinds": dict(sorted(kind_counts.items())),
        "required_kinds": sorted(set(args.require_kind)),
        "source_groups": len({record.source_group for record in records}),
        "vocal_with_verbatim_lyrics": sum(
            record.kind == "vocal" and record.verbatim is True for record in records
        ),
        "licenses": dict(
            sorted(Counter(record.license or "unknown" for record in records).items())
        ),
        "missing_files_allowed": args.allow_missing,
        "all_files_verified": not args.allow_missing,
    }
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
