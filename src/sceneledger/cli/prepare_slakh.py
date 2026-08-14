"""Build a leakage-safe instrumental music catalog from Slakh2100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.slakh import SLAKH_LICENSE, VALID_VARIANTS, convert_slakh_records
from sceneledger.data.source_catalog import prepare_source_catalog, write_source_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-prepare-slakh")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-variant", required=True, choices=sorted(VALID_VARIANTS))
    parser.add_argument("--min-instrument-classes", type=int, default=4)
    parser.add_argument(
        "--allow-voice-like-instruments",
        action="store_true",
        help="not recommended; permits choir/voice-like synthesizer patches",
    )
    parser.add_argument("--audit-per-kind", type=int, default=30)
    parser.add_argument("--min-per-split", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = convert_slakh_records(
        args.root,
        split_variant=args.split_variant,
        reject_voice_like=not args.allow_voice_like_instruments,
        min_instrument_classes=args.min_instrument_classes,
    )
    raw_catalog = output / "slakh_raw.jsonl"
    write_source_catalog(raw_catalog, records)
    prepared = output / "prepared"
    report = prepare_source_catalog(
        raw_catalog,
        prepared,
        audio_root=args.root,
        allowed_licenses={SLAKH_LICENSE},
        seed=args.seed,
        audit_per_kind=args.audit_per_kind,
        min_records_per_kind_per_split=args.min_per_split,
        min_groups_per_kind_per_split=args.min_per_split,
        min_caption_unique_fraction=0.25,
        required_kinds={"music"},
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "eligible_tracks": len(records),
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
