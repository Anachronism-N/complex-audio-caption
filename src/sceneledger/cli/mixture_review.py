"""Prepare and summarize blinded Rule-versus-LLM mixture reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.eval.mixture_review import (
    prepare_mixture_review,
    summarize_mixture_review,
    write_mixture_review,
)


def _prepare(args: argparse.Namespace) -> int:
    rows, metadata, key = prepare_mixture_review(
        rule_recipes_path=args.rule_recipes,
        rule_manifest_path=args.rule_manifest,
        rule_quality_report_path=args.rule_quality_report,
        rule_audio_base=args.rule_audio_base,
        llm_recipes_path=args.llm_recipes,
        llm_manifest_path=args.llm_manifest,
        llm_quality_report_path=args.llm_quality_report,
        llm_audio_base=args.llm_audio_base,
        package_dir=args.package_dir,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    write_mixture_review(
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
                "n_tasks": metadata["n_tasks"],
                "by_template": metadata["by_template"],
                "package_dir": str(Path(args.package_dir).resolve()),
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
    summary = summarize_mixture_review(
        review_csv_path=args.review_csv,
        metadata_path=args.metadata,
        key_path=args.key,
        min_plausibility_delta=args.min_plausibility_delta,
        max_safety_regression=args.max_safety_regression,
        max_sign_p=args.max_sign_p,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["go_for_scale"] else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-mixture-review")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="create a frozen anonymous A/B audio-and-stem package"
    )
    prepare.add_argument("--rule-recipes", required=True)
    prepare.add_argument("--rule-manifest", required=True)
    prepare.add_argument("--rule-quality-report", required=True)
    prepare.add_argument("--rule-audio-base", required=True)
    prepare.add_argument("--llm-recipes", required=True)
    prepare.add_argument("--llm-manifest", required=True)
    prepare.add_argument("--llm-quality-report", required=True)
    prepare.add_argument("--llm-audio-base", required=True)
    prepare.add_argument("--package-dir", required=True)
    prepare.add_argument("--sample-count", type=int, default=120)
    prepare.add_argument("--seed", default="sceneledger-mixture-planner-review-v1")
    prepare.add_argument("--output-csv", required=True)
    prepare.add_argument("--output-metadata", required=True)
    prepare.add_argument("--output-key", required=True)
    prepare.set_defaults(handler=_prepare)

    summarize = commands.add_parser(
        "summarize", help="validate completed sheets, unblind, and compute the scale gate"
    )
    summarize.add_argument(
        "--review-csv",
        action="append",
        required=True,
        help="completed sheet; repeat for independent reviewers",
    )
    summarize.add_argument("--metadata", required=True)
    summarize.add_argument("--key", required=True)
    summarize.add_argument("--min-plausibility-delta", type=float, default=0.25)
    summarize.add_argument("--max-safety-regression", type=float, default=0.10)
    summarize.add_argument("--max-sign-p", type=float, default=0.05)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(handler=_summarize)

    args = parser.parse_args(argv)
    if not 0.0 <= getattr(args, "max_sign_p", 0.05) <= 1.0:
        parser.error("--max-sign-p must be in [0, 1]")
    if getattr(args, "max_safety_regression", 0.10) < 0.0:
        parser.error("--max-safety-regression must be non-negative")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
