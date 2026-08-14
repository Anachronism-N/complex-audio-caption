"""Prepare and gate a LibriSpeech source catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.librispeech import LIBRISPEECH_LICENSE, convert_librispeech_records
from sceneledger.data.source_catalog import prepare_source_catalog, write_source_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-prepare-librispeech")
    parser.add_argument("--root", required=True, help="directory containing LibriSpeech subsets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--subset",
        action="append",
        default=[],
        help="subset to include; repeat (default: every supported subset found)",
    )
    parser.add_argument("--max-per-speaker", type=int, default=None)
    parser.add_argument("--audit-per-kind", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = convert_librispeech_records(
        args.root,
        subsets=set(args.subset) if args.subset else None,
        max_per_speaker=args.max_per_speaker,
    )
    raw_catalog = output / "librispeech_raw.jsonl"
    write_source_catalog(raw_catalog, records)
    prepared = output / "prepared"
    report = prepare_source_catalog(
        raw_catalog,
        prepared,
        audio_root=args.root,
        allowed_licenses={LIBRISPEECH_LICENSE},
        seed=args.seed,
        audit_per_kind=args.audit_per_kind,
        min_records_per_kind_per_split=4,
        min_groups_per_kind_per_split=4,
        min_caption_unique_fraction=0.95,
        required_kinds={"speech"},
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "n_records": len(records),
                "raw_catalog": str(raw_catalog),
                "report": str(prepared / "source_catalog_report.json"),
            },
            ensure_ascii=False,
        )
    )
    if not report["pass"]:
        for check in report.get("checks", []):
            if not check["pass"]:
                print(f"FAIL {check['name']}: {check['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
