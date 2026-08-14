"""Prepare and summarize the fail-closed human listening audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.experiment_data import (
    file_sha256,
    require_experiment_data_summary,
    require_split_manifest,
)
from sceneledger.data.human_audit import (
    build_human_audit,
    summarize_human_audit,
    write_human_audit,
)


def _prepare(args: argparse.Namespace) -> int:
    gate = require_experiment_data_summary(args.data_gate_summary, args.split_contract)
    contract = require_split_manifest(args.split_contract, args.expected_split, args.manifest)
    if gate["dataset_id"] != contract["dataset_id"]:
        raise ValueError("data gate and split contract dataset IDs differ")
    quality_item = gate["quality_reports"][args.expected_split]
    quality_path = Path(quality_item["path"])
    if file_sha256(quality_path) != quality_item["sha256"]:
        raise ValueError("quality report changed after the data gate")
    quality_report = json.loads(quality_path.read_text(encoding="utf-8"))

    rows, metadata = build_human_audit(
        args.manifest,
        quality_report,
        dataset_id=gate["dataset_id"],
        split=args.expected_split,
        per_template=args.per_template,
        max_violation_samples=args.max_violation_samples,
        seed=args.seed,
        all_samples=args.all_samples,
    )
    metadata["data_gate_summary_path"] = str(Path(args.data_gate_summary).resolve())
    metadata["data_gate_summary_sha256"] = file_sha256(args.data_gate_summary)
    metadata["split_contract_path"] = str(Path(args.split_contract).resolve())
    metadata["split_contract_sha256"] = file_sha256(args.split_contract)
    metadata["quality_report_path"] = str(quality_path.resolve())
    metadata["quality_report_sha256"] = file_sha256(quality_path)
    write_human_audit(rows, metadata, args.output_csv, args.output_metadata)
    print(
        json.dumps(
            {
                "audit_id": metadata["audit_id"],
                "dataset_id": metadata["dataset_id"],
                "n_tasks": metadata["n_tasks"],
                "by_template": metadata["by_template"],
                "output_csv": str(Path(args.output_csv).resolve()),
                "output_metadata": str(Path(args.output_metadata).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _prepare_standalone(args: argparse.Namespace) -> int:
    quality_path = Path(args.quality_report).resolve()
    quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality_report.get("pass") is not True or quality_report.get("failed_checks"):
        raise ValueError("standalone human audit requires a passed mixture-quality report")
    rows, metadata = build_human_audit(
        args.manifest,
        quality_report,
        dataset_id=args.dataset_id,
        split=args.split,
        per_template=args.per_template,
        max_violation_samples=args.max_violation_samples,
        seed=args.seed,
        all_samples=args.all_samples,
    )
    metadata["quality_report_path"] = str(quality_path)
    metadata["quality_report_sha256"] = file_sha256(quality_path)
    metadata["manifest_path"] = str(Path(args.manifest).resolve())
    metadata["manifest_sha256"] = file_sha256(args.manifest)
    write_human_audit(rows, metadata, args.output_csv, args.output_metadata)
    print(
        json.dumps(
            {
                "audit_id": metadata["audit_id"],
                "dataset_id": metadata["dataset_id"],
                "n_tasks": metadata["n_tasks"],
                "by_template": metadata["by_template"],
                "output_csv": str(Path(args.output_csv).resolve()),
                "output_metadata": str(Path(args.output_metadata).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _summarize(args: argparse.Namespace) -> int:
    summary = summarize_human_audit(
        args.review_csv,
        args.metadata,
        max_severe=args.max_severe,
        max_total_failures=args.max_total_failures,
        template_failure_threshold=args.template_failure_threshold,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-human-audit")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="create a frozen listening sheet")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--data-gate-summary", required=True)
    prepare.add_argument("--split-contract", required=True)
    prepare.add_argument("--expected-split", default="test", choices=("train", "val", "test"))
    prepare.add_argument("--per-template", type=int, default=5)
    prepare.add_argument("--max-violation-samples", type=int, default=20)
    prepare.add_argument("--seed", default=None)
    prepare.add_argument("--all-samples", action="store_true")
    prepare.add_argument("--output-csv", required=True)
    prepare.add_argument("--output-metadata", required=True)
    prepare.set_defaults(handler=_prepare)

    standalone = commands.add_parser(
        "prepare-standalone",
        help="create a frozen listening sheet for a passed single-split pilot",
    )
    standalone.add_argument("--manifest", required=True)
    standalone.add_argument("--quality-report", required=True)
    standalone.add_argument("--dataset-id", required=True)
    standalone.add_argument("--split", default="test", choices=("train", "val", "test"))
    standalone.add_argument("--per-template", type=int, default=5)
    standalone.add_argument("--max-violation-samples", type=int, default=20)
    standalone.add_argument("--seed", default=None)
    standalone.add_argument("--all-samples", action="store_true")
    standalone.add_argument("--output-csv", required=True)
    standalone.add_argument("--output-metadata", required=True)
    standalone.set_defaults(handler=_prepare_standalone)

    summarize = commands.add_parser("summarize", help="validate answers and build the human gate")
    summarize.add_argument("--review-csv", required=True)
    summarize.add_argument("--metadata", required=True)
    summarize.add_argument("--output", required=True)
    summarize.add_argument("--max-severe", type=int, default=2)
    summarize.add_argument("--max-total-failures", type=int, default=2)
    summarize.add_argument("--template-failure-threshold", type=int, default=2)
    summarize.set_defaults(handler=_summarize)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
