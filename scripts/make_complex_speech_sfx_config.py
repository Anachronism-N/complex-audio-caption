"""Build a server-local config for the first six-source complex-data anchor."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sceneledger.data.scene_recipes import (
    build_label_inventory,
    generate_rule_recipes,
    write_inventory,
    write_recipe_review,
    write_recipes,
)

try:  # direct script execution adds scripts/, tests import the namespace package
    from scripts.make_real_speech_sfx_pilot_config import (
        _passed_split_audit,
        _rms_ready_catalog,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by server CLI usage
    from make_real_speech_sfx_pilot_config import (  # type: ignore[no-redef]
        _passed_split_audit,
        _rms_ready_catalog,
    )


def _directory(value: str, description: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{description} is missing: {path}")
    return path


def _catalog_config(
    prepared_value: str,
    audio_root: Path,
    name: str,
    required_kinds: set[str],
    split: str,
) -> dict[str, object]:
    prepared = Path(prepared_value).expanduser().resolve()
    catalog = _rms_ready_catalog(
        prepared / f"{split}.jsonl", f"{name} {split} catalog"
    )
    audit = _passed_split_audit(
        prepared / "source_audit_report.json",
        f"{name} source audit",
        required_split=split,
        required_kinds=required_kinds,
        minimum_per_kind=10,
    )
    return {
        "catalog_path": str(catalog),
        "audio_root": str(audio_root),
        "audit_report_path": str(audit),
        "expected_split": split,
        "sampling_weight": 1.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", required=True)
    parser.add_argument("--librispeech-prepared", required=True)
    parser.add_argument("--esc50-audio-root", required=True)
    parser.add_argument("--esc50-prepared", required=True)
    parser.add_argument("--fsd50k-root", required=True)
    parser.add_argument("--fsd50k-prepared", required=True)
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--seed-base", type=int, default=2026081500)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.sample_count < 120:
        parser.error("--sample-count must be at least 120 for the registered gate")

    libri_root = _directory(args.librispeech_root, "LibriSpeech root")
    esc_root = _directory(args.esc50_audio_root, "ESC-50 audio root")
    fsd_root = _directory(args.fsd50k_root, "FSD50K root")
    catalogs = [
        _catalog_config(
            args.librispeech_prepared,
            libri_root,
            "LibriSpeech",
            {"speech"},
            args.split,
        ),
        _catalog_config(
            args.esc50_prepared,
            esc_root,
            "ESC-50",
            {"sfx", "ambience"},
            args.split,
        ),
        _catalog_config(
            args.fsd50k_prepared,
            fsd_root,
            "FSD50K",
            {"sfx", "ambience"},
            args.split,
        ),
    ]
    output = Path(args.output).expanduser().resolve()
    inventory_path = output.with_suffix(".inventory.json")
    recipes_path = output.with_suffix(".rule_recipes.jsonl")
    review_path = output.with_suffix(".rule_recipe_review.csv")
    artifacts = (output, inventory_path, recipes_path, review_path)
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite complex artifacts: {existing}")

    inventory = build_label_inventory(
        [str(item["catalog_path"]) for item in catalogs]
    )
    recipes = generate_rule_recipes(
        inventory,
        count=args.sample_count,
        seed=args.seed_base,
        template_weights={"multi_speaker_ambient_events": 1.0},
        recipe_prefix=f"complex_{args.split}_rule",
        strategy="keyword",
    )
    write_inventory(inventory_path, inventory)
    write_recipes(recipes_path, recipes)
    write_recipe_review(review_path, recipes)

    config = {
        "pool": {"kind": "catalog_set", "catalogs": catalogs},
        "sampler": {
            "sample_rate": 16000,
            "duration_range": [12.0, 18.0],
            "template_duration_ranges": {
                "multi_speaker_ambient_events": [12.0, 18.0]
            },
            "target_active_rms_dbfs_by_kind": {
                "speech": [-22.0, -20.0],
                "sfx": [-32.0, -28.0],
                "ambience": [-40.0, -36.0],
            },
            "max_abs_source_gain_db": 24.0,
            "foreground_onset_fraction_range": [0.02, 0.75],
            "t60_range": [0.15, 0.65],
            "echo_delay_ms_range": [90, 260],
            "echo_atten_db_range": [-18.0, -8.0],
            "merge_threshold_range": [0.1, 0.3],
            "resolutions": [0.1],
            "styles": ["detailed"],
            "loop_background_to_scene": True,
            "stable_unique_source_ids": True,
            "random_crop_backgrounds": True,
            "fade_in_range_by_kind": {
                "speech": [0.005, 0.02],
                "sfx": [0.002, 0.015],
                "ambience": [0.3, 1.0],
            },
            "fade_out_range_by_kind": {
                "speech": [0.005, 0.03],
                "sfx": [0.003, 0.03],
                "ambience": [0.5, 1.5],
            },
            # One room identity/T60 per scene.  Individual sources retain
            # deterministic source-position realizations inside that room.
            "shared_room_probability": 0.65,
            "p_rir": 0.0,
            "p_echo": 0.15,
            # Mild production-style ducking is an explicit condition and is
            # applied to saved stems, so it can be ablated exactly.
            "ducking_probability": 0.25,
            "ducking_depth_db_range": [2.0, 4.0],
        },
        "render": {
            "sample_count": args.sample_count,
            "scene_id_prefix": f"complex_speech_sfx_{args.split}",
            "recipe_plan_path": str(recipes_path),
            "recipe_inventory_path": str(inventory_path),
        },
        "recipe_experiment": {
            "proposal_source": "rules:keyword_v1",
            "human_review_path": str(review_path),
            "llm_recipes_deferred_until_rule_anchor_passes": True,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"config={output}")
    print(f"inventory={inventory_path}")
    print(f"rule_recipes={recipes_path}")
    print(f"recipe_review={review_path}")
    print("source_audits=passed")
    print(f"split={args.split}")
    print(
        "next=bash scripts/run_complex_speech_sfx_pilot.sh "
        f"{output} /new/output/directory"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
