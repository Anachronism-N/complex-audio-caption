from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from jsonschema import Draft202012Validator

from .data.carc import build_exact_carc
from .data.datasets import download_datasets
from .data.manifest import load_source_manifest, write_source_manifest
from .data.organize import assign_group_splits, build_source_manifest
from .data.renderer import render_tac_dataset
from .data.validate import validate_rendered_dataset, validate_source_manifest
from .integrations.moss import MossInferenceAdapter, write_moss_sft
from .metrics import evaluate_corpus
from .serialization import parse_tagged_caption, serialize_tagged_caption
from .types import Ledger, read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sceneledger")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate canonical ledger JSONL")
    validate.add_argument("input")
    validate.add_argument("--schema", default="schemas/track_event_ledger.schema.json")

    validate_sources = commands.add_parser(
        "validate-sources", help="Validate source files and splits"
    )
    validate_sources.add_argument("input")
    validate_sources.add_argument("--verify-hashes", action="store_true")

    validate_render = commands.add_parser(
        "validate-render", help="Validate rendered stems and labels"
    )
    validate_render.add_argument("input")
    validate_render.add_argument("--tolerance", type=float, default=2e-4)

    serialize = commands.add_parser("serialize", help="Serialize ledger JSONL into tagged captions")
    serialize.add_argument("input")
    serialize.add_argument("output")

    parse = commands.add_parser("parse", help="Parse one tagged caption into ledger JSON")
    parse.add_argument("input")
    parse.add_argument("output")
    parse.add_argument("--sample-id", required=True)
    parse.add_argument("--duration-sec", type=float, required=True)
    parse.add_argument("--non-strict", action="store_true")

    evaluate = commands.add_parser("evaluate", help="Evaluate prediction ledgers")
    evaluate.add_argument("reference")
    evaluate.add_argument("prediction")
    evaluate.add_argument("--output")

    download = commands.add_parser("download", help="Download registered public datasets")
    download.add_argument("--registry", default="configs/data/datasets.yaml")
    download.add_argument("--output-root", required=True)
    download.add_argument("--dataset", action="append", required=True)
    download.add_argument("--accept-license", action="append", default=[])
    download.add_argument("--extract", action="store_true")

    organize = commands.add_parser("organize", help="Index authorized audio into a source manifest")
    organize.add_argument("--input-root", required=True)
    organize.add_argument("--output", required=True)
    organize.add_argument("--type", choices=["speech", "lys", "music", "sfx", "ambience"])
    organize.add_argument("--metadata")
    organize.add_argument("--license")
    organize.add_argument("--assign-splits", action="store_true")
    organize.add_argument("--train-ratio", type=float, default=0.9)
    organize.add_argument("--validation-ratio", type=float, default=0.05)
    organize.add_argument("--seed", type=int, default=20260808)

    split = commands.add_parser("split", help="Assign leakage-safe deterministic group splits")
    split.add_argument("input")
    split.add_argument("output")
    split.add_argument("--train-ratio", type=float, default=0.9)
    split.add_argument("--validation-ratio", type=float, default=0.05)
    split.add_argument("--seed", type=int, default=20260808)

    render = commands.add_parser("render", help="Render TAC-mini mixtures")
    render.add_argument("--sources", required=True)
    render.add_argument("--config", default="configs/data/tac_mini.yaml")
    render.add_argument("--output", required=True)

    carc = commands.add_parser("carc", help="Build Exact-CARC add/remove/shift groups")
    carc.add_argument("--backgrounds", required=True)
    carc.add_argument("--sources", required=True)
    carc.add_argument("--config", default="configs/data/exact_carc.yaml")
    carc.add_argument("--output", required=True)

    moss_sft = commands.add_parser("moss-sft", help="Convert ledgers to official MOSS SFT JSONL")
    moss_sft.add_argument("--ledgers", required=True)
    moss_sft.add_argument("--render-manifest", required=True)
    moss_sft.add_argument("--data-root")
    moss_sft.add_argument("--output", required=True)
    moss_sft.add_argument("--prompt")

    moss_infer = commands.add_parser("moss-infer", help="Run official MOSS-Audio inference")
    moss_infer.add_argument("--upstream-root", required=True)
    moss_infer.add_argument("--model-path", required=True)
    moss_infer.add_argument("--audio", required=True)
    moss_infer.add_argument("--prompt", default="Describe this audio.")
    moss_infer.add_argument("--output")
    moss_infer.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        count = _validate_ledgers(args.input, args.schema)
        print(json.dumps({"valid": count}, ensure_ascii=False))
    elif args.command == "validate-sources":
        result = validate_source_manifest(args.input, verify_hashes=args.verify_hashes)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["valid"]:
            return 2
    elif args.command == "validate-render":
        result = validate_rendered_dataset(args.input, tolerance=args.tolerance)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["valid"]:
            return 2
    elif args.command == "serialize":
        rows = [
            {"sample_id": ledger.sample_id, "caption": serialize_tagged_caption(ledger)}
            for ledger in read_jsonl(args.input)
        ]
        _write_rows(args.output, rows)
        print(json.dumps({"serialized": len(rows), "output": args.output}))
    elif args.command == "parse":
        caption = Path(args.input).read_text(encoding="utf-8")
        ledger = parse_tagged_caption(
            caption,
            sample_id=args.sample_id,
            duration_sec=args.duration_sec,
            strict=not args.non_strict,
        )
        Path(args.output).write_text(
            json.dumps(ledger.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif args.command == "evaluate":
        metrics = evaluate_corpus(read_jsonl(args.reference), read_jsonl(args.prediction)).to_dict()
        rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered)
    elif args.command == "download":
        result = download_datasets(
            args.registry,
            args.dataset,
            args.output_root,
            accepted_licenses=set(args.accept_license),
            extract=args.extract,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "organize":
        records = build_source_manifest(
            args.input_root,
            args.output,
            default_type=args.type,
            metadata_path=args.metadata,
            license_name=args.license,
        )
        if args.assign_splits:
            records = assign_group_splits(
                records,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                seed=args.seed,
            )
            write_source_manifest(args.output, records)
        print(json.dumps({"indexed": len(records), "output": args.output}))
    elif args.command == "split":
        records = assign_group_splits(
            load_source_manifest(args.input),
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        write_source_manifest(args.output, records)
        print(json.dumps({"records": len(records), "output": args.output}))
    elif args.command == "render":
        print(json.dumps(render_tac_dataset(args.sources, args.config, args.output), indent=2))
    elif args.command == "carc":
        print(
            json.dumps(
                build_exact_carc(args.backgrounds, args.sources, args.config, args.output), indent=2
            )
        )
    elif args.command == "moss-sft":
        data_root = (
            Path(args.data_root).resolve()
            if args.data_root
            else Path(args.render_manifest).resolve().parent
        )
        audio_paths = {}
        for row in _read_rows(args.render_manifest):
            audio_path = Path(row["mixture_path"])
            audio_paths[row["sample_id"]] = (
                audio_path if audio_path.is_absolute() else data_root / audio_path
            )
        kwargs = {}
        if args.prompt:
            kwargs["prompt"] = args.prompt
        count = write_moss_sft(read_jsonl(args.ledgers), audio_paths, args.output, **kwargs)
        print(json.dumps({"samples": count, "output": args.output}))
    elif args.command == "moss-infer":
        adapter = MossInferenceAdapter(args.upstream_root, args.model_path, device=args.device)
        result = adapter.generate(args.audio, args.prompt)
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
        print(result)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


def _validate_ledgers(input_path: str, schema_path: str) -> int:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    count = 0
    for value in _read_rows(input_path):
        errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
        if errors:
            locations = [f"{list(error.path)}: {error.message}" for error in errors[:10]]
            raise ValueError("Schema validation failed: " + "; ".join(locations))
        Ledger.from_dict(value).validate()
        count += 1
    return count


def _read_rows(path: str | Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def _write_rows(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
