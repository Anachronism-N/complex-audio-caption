"""Build a licensed, auditable FSD50K SFX/ambience source catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.fsd50k import convert_fsd50k_records
from sceneledger.data.source_catalog import prepare_source_catalog, write_source_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-prepare-fsd50k")
    parser.add_argument("--root", required=True, help="extracted FSD50K root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-license",
        action="append",
        choices=("CC0-1.0", "CC BY 3.0", "CC BY-NC 3.0", "CC Sampling+ 1.0"),
        default=[],
    )
    parser.add_argument("--audit-per-kind", type=int, default=30)
    parser.add_argument("--min-per-kind-per-split", type=int, default=50)
    args = parser.parse_args(argv)
    if not args.allow_license:
        parser.error("repeat --allow-license for every accepted per-clip license")

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = convert_fsd50k_records(
        args.root,
        allowed_licenses=set(args.allow_license),
        include_eval=True,
    )
    raw_catalog = output / "fsd50k_raw.jsonl"
    write_source_catalog(raw_catalog, records)
    prepared = output / "prepared"
    report = prepare_source_catalog(
        raw_catalog,
        prepared,
        audio_root=args.root,
        allowed_licenses=set(args.allow_license),
        audit_per_kind=args.audit_per_kind,
        min_records_per_kind_per_split=args.min_per_kind_per_split,
        min_groups_per_kind_per_split=args.min_per_kind_per_split,
        min_caption_unique_fraction=0.005,
        required_kinds={"sfx", "ambience"},
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "eligible_records": len(records),
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
