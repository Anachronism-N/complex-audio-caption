"""Create frozen train/val/test specifications for the first real complex anchor.

This script never renders audio.  It binds each fold to the corresponding
prepared source-catalog split and creates independent inventories, deterministic
rule recipes, and human recipe-review sheets.  Rendering is a separate,
fail-closed step after all three review sheets are complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.make_complex_speech_sfx_config import (
        _catalog_config,
        _directory,
    )
    from scripts.make_complex_speech_sfx_config import (
        main as make_split_config,
    )
except ModuleNotFoundError:  # pragma: no cover - direct server execution
    from make_complex_speech_sfx_config import (  # type: ignore[no-redef]
        _catalog_config,
        _directory,
    )
    from make_complex_speech_sfx_config import (
        main as make_split_config,
    )


SPLITS = ("train", "val", "test")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", required=True)
    parser.add_argument("--librispeech-prepared", required=True)
    parser.add_argument("--esc50-audio-root", required=True)
    parser.add_argument("--esc50-prepared", required=True)
    parser.add_argument("--fsd50k-root", required=True)
    parser.add_argument("--fsd50k-prepared", required=True)
    parser.add_argument("--train-count", type=int, default=120)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--test-count", type=int, default=120)
    parser.add_argument("--seed-base", type=int, default=2026081600)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    too_small = {split: count for split, count in counts.items() if count < 120}
    if too_small:
        parser.error(
            "every fold must contain at least 120 scenes for the registered "
            f"complexity profile: {too_small}"
        )

    roots = {
        "librispeech": _directory(args.librispeech_root, "LibriSpeech root"),
        "esc50": _directory(args.esc50_audio_root, "ESC-50 audio root"),
        "fsd50k": _directory(args.fsd50k_root, "FSD50K root"),
    }

    # Validate every source fold before creating any experiment artifact.
    for split in SPLITS:
        _catalog_config(
            args.librispeech_prepared,
            roots["librispeech"],
            "LibriSpeech",
            {"speech"},
            split,
        )
        _catalog_config(
            args.esc50_prepared,
            roots["esc50"],
            "ESC-50",
            {"sfx", "ambience"},
            split,
        )
        _catalog_config(
            args.fsd50k_prepared,
            roots["fsd50k"],
            "FSD50K",
            {"sfx", "ambience"},
            split,
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse experiment specification directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    artifacts: dict[str, dict[str, str | int]] = {}
    for index, split in enumerate(SPLITS):
        config_path = output_dir / f"{split}.yaml"
        make_split_config(
            [
                "--librispeech-root",
                str(roots["librispeech"]),
                "--librispeech-prepared",
                args.librispeech_prepared,
                "--esc50-audio-root",
                str(roots["esc50"]),
                "--esc50-prepared",
                args.esc50_prepared,
                "--fsd50k-root",
                str(roots["fsd50k"]),
                "--fsd50k-prepared",
                args.fsd50k_prepared,
                "--sample-count",
                str(counts[split]),
                "--seed-base",
                str(args.seed_base + index * 1_000_000),
                "--split",
                split,
                "--output",
                str(config_path),
            ]
        )
        artifacts[split] = {
            "sample_count": counts[split],
            "config": str(config_path),
            "inventory": str(config_path.with_suffix(".inventory.json")),
            "recipes": str(config_path.with_suffix(".rule_recipes.jsonl")),
            "recipe_review": str(
                config_path.with_suffix(".rule_recipe_review.csv")
            ),
        }

    summary = {
        "schema_version": "sceneledger.complex_experiment_spec.v1",
        "seed_base": args.seed_base,
        "source_splits_required": list(SPLITS),
        "folds": artifacts,
        "next": (
            "Complete all three *.rule_recipe_review.csv files, then run "
            "scripts/run_complex_speech_sfx_experiment.sh SPEC_DIR OUTPUT_DIR"
        ),
    }
    summary_path = output_dir / "experiment_spec.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"experiment_spec={summary_path}")
    print("recipe_review_required=train,val,test")
    print(f"next=bash scripts/run_complex_speech_sfx_experiment.sh {output_dir} OUTPUT_DIR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
