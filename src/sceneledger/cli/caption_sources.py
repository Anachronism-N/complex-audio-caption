"""CLI for the frozen real-source MOSS zero-shot anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.source_captioning import (
    build_source_caption_plan,
    run_source_caption_plan,
    validate_source_caption_audit,
    write_source_caption_audit,
)


def _write_report(path: str | Path, report: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-caption-sources")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("make-plan", help="freeze a balanced source/prompt plan")
    plan.add_argument("--catalog", required=True, help="prepared all.jsonl")
    plan.add_argument("--audio-root", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--per-label", type=int, default=1)
    plan.add_argument("--seed", type=int, default=20260813)
    plan.add_argument(
        "--prompt-mode",
        action="append",
        choices=("semantic", "structured"),
        default=[],
        help="repeat to select modes; default runs the paired semantic+structured test",
    )

    run = commands.add_parser("run", help="run MOSS and persist every raw generation")
    run.add_argument("--plan", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--max-new-tokens", type=int, default=1024)
    run.add_argument("--resume", action="store_true")

    audit = commands.add_parser("make-audit", help="create the human listening sheet")
    audit.add_argument("--results", required=True)
    audit.add_argument("--output", required=True)

    validate = commands.add_parser("validate-audit", help="produce a frozen go/no-go report")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--results", required=True)
    validate.add_argument("--audit", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--min-label-correct", type=float, default=0.70)
    validate.add_argument("--min-coverage", type=float, default=0.70)
    validate.add_argument("--min-hallucination-free", type=float, default=0.80)
    validate.add_argument("--min-structured-format", type=float, default=0.50)
    validate.add_argument("--min-temporal-supported", type=float, default=0.70)
    validate.add_argument("--min-structured-temporal-claims", type=float, default=0.50)
    args = parser.parse_args(argv)

    if args.command == "make-plan":
        result = build_source_caption_plan(
            args.catalog,
            args.audio_root,
            args.output,
            per_label=args.per_label,
            seed=args.seed,
            prompt_modes=tuple(args.prompt_mode or ("semantic", "structured")),
        )
        print(json.dumps({key: result[key] for key in ("n_labels", "n_sources", "n_generations")}, ensure_ascii=False))
        return 0

    if args.command == "run":
        from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig

        config = MossAdapterConfig(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
        )
        report = run_source_caption_plan(
            args.plan,
            args.output,
            MossAdapter(config),
            resume=args.resume,
            model_metadata={
                "backend": "moss",
                "model_path": args.model_path,
                "device": args.device,
                "dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens,
                "greedy": True,
            },
        )
        _write_report(args.report, report)
        print(json.dumps({"pass": report["pass"], "report": args.report}, ensure_ascii=False))
        return 0 if report["pass"] else 1

    if args.command == "make-audit":
        count = write_source_caption_audit(args.results, args.output)
        print(json.dumps({"rows": count, "audit": args.output}, ensure_ascii=False))
        return 0

    report = validate_source_caption_audit(
        args.plan,
        args.results,
        args.audit,
        args.output,
        min_label_correct=args.min_label_correct,
        min_coverage=args.min_coverage,
        min_hallucination_free=args.min_hallucination_free,
        min_structured_format=args.min_structured_format,
        min_temporal_supported=args.min_temporal_supported,
        min_structured_temporal_claims=args.min_structured_temporal_claims,
    )
    print(json.dumps({"pass": report["pass"], "rates": report["rates"], "report": args.output}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
