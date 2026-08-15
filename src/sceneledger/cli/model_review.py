"""Prepare and summarize blinded zero-shot versus tuned listening reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.eval.model_review import (
    prepare_model_review,
    summarize_model_review,
    write_model_review,
)


def _prepare(args: argparse.Namespace) -> int:
    rows, metadata, key = prepare_model_review(
        manifest_path=args.manifest,
        audio_base=args.audio_base,
        zero_predictions_path=args.zero_predictions,
        zero_inference_report_path=args.zero_inference_report,
        tuned_predictions_path=args.tuned_predictions,
        tuned_inference_report_path=args.tuned_inference_report,
        validity_audit_path=args.validity_audit,
        split_contract_path=args.split_contract,
        data_gate_summary_path=args.data_gate_summary,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    write_model_review(
        rows,
        metadata,
        key,
        csv_path=args.output_csv,
        metadata_path=args.output_metadata,
        key_path=args.output_key,
    )
    print(
        json.dumps(
            {
                "review_id": metadata["review_id"],
                "dataset_id": metadata["dataset_id"],
                "n_tasks": metadata["n_tasks"],
                "by_template": metadata["by_template"],
                "output_csv": str(Path(args.output_csv).resolve()),
                "output_metadata": str(Path(args.output_metadata).resolve()),
                "output_key": str(Path(args.output_key).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _summarize(args: argparse.Namespace) -> int:
    summary = summarize_model_review(
        review_csv_path=args.review_csv,
        metadata_path=args.metadata,
        key_path=args.key,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-model-review")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="create blinded A/B review tasks")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--audio-base", required=True)
    prepare.add_argument("--zero-predictions", required=True)
    prepare.add_argument("--zero-inference-report", required=True)
    prepare.add_argument("--tuned-predictions", required=True)
    prepare.add_argument("--tuned-inference-report", required=True)
    prepare.add_argument("--validity-audit", required=True)
    prepare.add_argument("--split-contract", required=True)
    prepare.add_argument("--data-gate-summary", required=True)
    prepare.add_argument("--sample-count", type=int, default=60)
    prepare.add_argument("--seed", default="sceneledger-model-review-v1")
    prepare.add_argument("--output-csv", required=True)
    prepare.add_argument("--output-metadata", required=True)
    prepare.add_argument("--output-key", required=True)
    prepare.set_defaults(handler=_prepare)

    summarize = commands.add_parser("summarize", help="validate and unblind a review")
    summarize.add_argument(
        "--review-csv",
        action="append",
        required=True,
        help="completed sheet; repeat for independent reviewers",
    )
    summarize.add_argument("--metadata", required=True)
    summarize.add_argument("--key", required=True)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(handler=_summarize)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
