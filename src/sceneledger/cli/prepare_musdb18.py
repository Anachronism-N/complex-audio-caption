"""Build an auditable music/vocal catalog from locally decoded MUSDB18 stems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.musdb18 import convert_musdb18_records
from sceneledger.data.source_catalog import prepare_source_catalog, write_source_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-prepare-musdb18")
    parser.add_argument("--root", required=True)
    parser.add_argument("--tracklist", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-license",
        action="append",
        default=[],
        help="exact per-track license; repeat as needed",
    )
    parser.add_argument("--audit-per-kind", type=int, default=30)
    args = parser.parse_args(argv)
    if not args.allow_license:
        parser.error("at least one --allow-license is required")

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = convert_musdb18_records(
        args.root,
        tracklist_path=args.tracklist,
        allowed_licenses=set(args.allow_license),
    )
    raw_catalog = output / "musdb18_raw.jsonl"
    write_source_catalog(raw_catalog, records)
    prepared = output / "prepared"
    report = prepare_source_catalog(
        raw_catalog,
        prepared,
        audio_root=args.root,
        allowed_licenses=set(args.allow_license),
        audit_per_kind=args.audit_per_kind,
        min_records_per_kind_per_split=4,
        min_groups_per_kind_per_split=4,
        min_caption_unique_fraction=0.05,
        required_kinds={"music", "vocal"},
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "eligible_stems": len(records),
                "raw_catalog": str(raw_catalog),
                "report": str(prepared / "source_catalog_report.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
