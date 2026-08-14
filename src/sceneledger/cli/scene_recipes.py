"""Build, validate and review constrained rule/LLM scene recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.scene_recipes import (
    build_label_inventory,
    compare_recipe_sets,
    compile_llm_responses,
    export_llm_tasks,
    generate_rule_recipes,
    read_inventory,
    read_recipes,
    recipe_summary,
    validate_recipe_review,
    validate_recipes,
    write_inventory,
    write_recipe_review,
    write_recipes,
)


def _weight(value: str) -> tuple[str, float]:
    name, separator, raw_weight = value.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError("template weight must be TEMPLATE=WEIGHT")
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("template weight must be numeric") from exc
    if weight <= 0:
        raise argparse.ArgumentTypeError("template weight must be positive")
    return name.strip(), weight


def _weights(values: list[tuple[str, float]]) -> dict[str, float]:
    if not values:
        return {"speech_with_sfx": 1.0, "speech_ambience_sfx": 1.0}
    output = dict(values)
    if len(output) != len(values):
        raise ValueError("each template weight may be specified only once")
    return output


def _write_report(path: str | None, report: dict[str, object]) -> None:
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-recipes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--catalog", action="append", required=True)
    inventory_parser.add_argument("--output", required=True)

    rules_parser = subparsers.add_parser("rules")
    rules_parser.add_argument("--inventory", required=True)
    rules_parser.add_argument("--count", type=int, required=True)
    rules_parser.add_argument("--seed", type=int, default=20260814)
    rules_parser.add_argument("--template-weight", action="append", type=_weight, default=[])
    rules_parser.add_argument("--prefix", default="rule")
    rules_parser.add_argument(
        "--strategy", choices=("keyword", "uniform"), default="keyword"
    )
    rules_parser.add_argument("--output", required=True)
    rules_parser.add_argument("--report", default=None)

    tasks_parser = subparsers.add_parser("llm-tasks")
    tasks_parser.add_argument("--inventory", required=True)
    tasks_parser.add_argument("--count", type=int, required=True)
    tasks_parser.add_argument("--seed", type=int, default=20260814)
    tasks_parser.add_argument("--template-weight", action="append", type=_weight, default=[])
    tasks_parser.add_argument("--max-labels-per-kind", type=int, default=120)
    tasks_parser.add_argument("--output", required=True)

    compile_parser = subparsers.add_parser("compile-llm")
    compile_parser.add_argument("--tasks", required=True)
    compile_parser.add_argument("--responses", required=True)
    compile_parser.add_argument("--model-name", required=True)
    compile_parser.add_argument("--inventory", required=True)
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument("--report", default=None)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--recipes", required=True)
    validate_parser.add_argument("--inventory", required=True)
    validate_parser.add_argument("--output", default=None)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--recipes", required=True)
    summarize_parser.add_argument("--output", default=None)

    review_parser = subparsers.add_parser("review-sheet")
    review_parser.add_argument("--recipes", required=True)
    review_parser.add_argument("--output", required=True)

    validate_review_parser = subparsers.add_parser("validate-review")
    validate_review_parser.add_argument("--recipes", required=True)
    validate_review_parser.add_argument("--review-csv", required=True)
    validate_review_parser.add_argument("--min-pass-rate", type=float, default=0.90)
    validate_review_parser.add_argument("--output", default=None)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--output", default=None)

    args = parser.parse_args(argv)
    if args.command == "inventory":
        inventory = build_label_inventory(args.catalog)
        write_inventory(args.output, inventory)
        print(args.output)
        return 0
    if args.command == "rules":
        inventory = read_inventory(args.inventory)
        recipes = generate_rule_recipes(
            inventory,
            count=args.count,
            seed=args.seed,
            template_weights=_weights(args.template_weight),
            recipe_prefix=args.prefix,
            strategy=args.strategy,
        )
        write_recipes(args.output, recipes)
        _write_report(args.report, validate_recipes(recipes, inventory))
        return 0
    if args.command == "llm-tasks":
        inventory = read_inventory(args.inventory)
        export_llm_tasks(
            inventory,
            count=args.count,
            seed=args.seed,
            template_weights=_weights(args.template_weight),
            output_path=args.output,
            max_labels_per_kind=args.max_labels_per_kind,
        )
        print(args.output)
        return 0
    if args.command == "compile-llm":
        inventory = read_inventory(args.inventory)
        recipes = compile_llm_responses(
            args.tasks,
            args.responses,
            output_path=args.output,
            model_name=args.model_name,
        )
        _write_report(args.report, validate_recipes(recipes, inventory))
        return 0
    if args.command == "validate":
        report = validate_recipes(
            read_recipes(args.recipes), read_inventory(args.inventory)
        )
        _write_report(args.output, report)
        return 0
    if args.command == "summarize":
        _write_report(args.output, recipe_summary(read_recipes(args.recipes)))
        return 0
    if args.command == "review-sheet":
        write_recipe_review(args.output, read_recipes(args.recipes))
        print(args.output)
        return 0
    if args.command == "validate-review":
        report = validate_recipe_review(
            args.review_csv,
            read_recipes(args.recipes),
            min_pass_rate=args.min_pass_rate,
        )
        _write_report(args.output, report)
        return 0 if report["pass"] else 1
    if args.command == "compare":
        report = compare_recipe_sets(
            read_recipes(args.left), read_recipes(args.right)
        )
        _write_report(args.output, report)
        return 0 if report["pass"] else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
