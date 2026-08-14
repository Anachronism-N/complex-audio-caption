"""CLI for normalizing and gating real single-source audio catalogs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.source_catalog import prepare_source_catalog, validate_source_audit


def _ratios(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratios must be comma-separated numbers") from exc
    if len(parts) != 3 or any(item <= 0 for item in parts):
        raise argparse.ArgumentTypeError("ratios must be three positive values, e.g. 0.8,0.1,0.1")
    return parts  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sceneledger-prepare-sources",
        description="Probe, fingerprint, group-split and audit a real source catalog.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="probe and split a raw source catalog")
    prepare.add_argument("--input", required=True, help="raw source catalog JSONL")
    prepare.add_argument("--audio-root", required=True, help="root for relative audio_path values")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument(
        "--allow-license",
        action="append",
        default=[],
        help="exact allowlisted license string; repeat for multiple values (fail-closed)",
    )
    prepare.add_argument("--split-ratios", type=_ratios, default=(0.8, 0.1, 0.1))
    prepare.add_argument("--seed", type=int, default=20260813)
    prepare.add_argument("--fingerprint-window-sec", type=float, default=10.0)
    prepare.add_argument("--min-duration-sec", type=float, default=0.2)
    prepare.add_argument("--min-rms-dbfs", type=float, default=-55.0)
    prepare.add_argument("--max-clipped-fraction", type=float, default=0.01)
    prepare.add_argument("--audit-per-kind", type=int, default=10)
    prepare.add_argument("--min-records-per-kind-per-split", type=int, default=4)
    prepare.add_argument("--min-groups-per-kind-per-split", type=int, default=4)
    prepare.add_argument("--min-caption-unique-fraction", type=float, default=0.5)
    prepare.add_argument(
        "--required-kind",
        action="append",
        choices=("speech", "vocal", "music", "sfx", "ambience"),
        default=[],
        help="kind required in every split; repeat as needed (default: all five kinds)",
    )

    audit = subparsers.add_parser("validate-audit", help="validate completed human source review")
    audit.add_argument("--preparation-report", required=True)
    audit.add_argument("--audit-csv", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--min-per-kind", type=int, default=10)
    audit.add_argument("--min-pass-rate", type=float, default=0.9)
    audit.add_argument(
        "--required-split",
        action="append",
        choices=("train", "val", "test"),
        default=[],
        help="require coverage and pass rate for this split; repeat as needed",
    )
    audit.add_argument("--min-per-kind-per-required-split", type=int, default=3)
    args = parser.parse_args(argv)

    if args.command == "validate-audit":
        report = validate_source_audit(
            args.preparation_report,
            args.audit_csv,
            args.output,
            min_per_kind=args.min_per_kind,
            min_pass_rate=args.min_pass_rate,
            required_splits=set(args.required_split),
            min_per_kind_per_required_split=args.min_per_kind_per_required_split,
        )
        print(json.dumps({"pass": report["pass"], "report": args.output}, ensure_ascii=False))
        return 0 if report["pass"] else 1

    if not args.allow_license:
        prepare.error("at least one --allow-license is required; licensing is fail-closed")
    report = prepare_source_catalog(
        args.input,
        args.output_dir,
        audio_root=args.audio_root,
        allowed_licenses=set(args.allow_license),
        split_ratios=args.split_ratios,
        seed=args.seed,
        fingerprint_window_sec=args.fingerprint_window_sec,
        min_duration_sec=args.min_duration_sec,
        min_rms_dbfs=args.min_rms_dbfs,
        max_clipped_fraction=args.max_clipped_fraction,
        audit_per_kind=args.audit_per_kind,
        min_records_per_kind_per_split=args.min_records_per_kind_per_split,
        min_groups_per_kind_per_split=args.min_groups_per_kind_per_split,
        min_caption_unique_fraction=args.min_caption_unique_fraction,
        required_kinds=set(args.required_kind) if args.required_kind else None,
    )
    report_path = Path(args.output_dir) / "source_catalog_report.json"
    print(json.dumps({"pass": report["pass"], "report": str(report_path)}, ensure_ascii=False))
    if not report["pass"]:
        for check in report.get("checks", []):
            if not check["pass"]:
                print(f"FAIL {check['name']}: {check['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
