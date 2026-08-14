"""Build a leakage-safe ESC-50 source catalog from official metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.esc50 import (
    ESC10_LICENSE,
    ESC50_NONCOMMERCIAL_LICENSE,
    convert_esc50_records,
    read_esc50_metadata,
)
from sceneledger.data.source_catalog import prepare_source_catalog, write_source_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-prepare-esc50")
    parser.add_argument("--metadata", required=True, help="meta/esc50.csv, parquet file, or directory")
    parser.add_argument("--audio-root", required=True, help="directory containing ESC-50 WAV files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--audit-per-kind", type=int, default=10)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="development only: permit a subset instead of the official 2000 clips",
    )
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = convert_esc50_records(
        read_esc50_metadata(args.metadata), require_complete=not args.allow_incomplete
    )
    raw_catalog = output / "esc50_raw.jsonl"
    write_source_catalog(raw_catalog, records)
    prepared = output / "prepared"
    report = prepare_source_catalog(
        raw_catalog,
        prepared,
        audio_root=args.audio_root,
        allowed_licenses={ESC10_LICENSE, ESC50_NONCOMMERCIAL_LICENSE},
        seed=args.seed,
        audit_per_kind=args.audit_per_kind,
        min_records_per_kind_per_split=4,
        min_groups_per_kind_per_split=4,
        min_caption_unique_fraction=0.02,
        required_kinds={"sfx", "ambience"},
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
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
