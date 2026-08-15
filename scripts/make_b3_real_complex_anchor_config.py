"""Bind B3 training to a passed real-complex experiment contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sceneledger.data.experiment_data import (
    require_experiment_data_summary,
    require_split_manifest,
)
from sceneledger.data.human_audit import require_human_audit_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.steps < 100:
        parser.error("--steps must be at least 100 for the registered anchor")

    root = Path(args.experiment_root).expanduser().resolve()
    gate = root / "gate"
    manifest = root / "train" / "manifest.jsonl"
    split_contract = gate / "split_contract.json"
    data_summary = gate / "experiment_data_summary.json"
    human_summary = gate / "human_audit_summary.json"
    data_gate = require_experiment_data_summary(data_summary, split_contract)
    require_split_manifest(split_contract, "train", manifest)
    require_human_audit_summary(
        human_summary, expected_dataset_id=data_gate["dataset_id"]
    )

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MOSS model path is missing: {model_path}")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite model config: {output}")

    config = {
        "model": {
            "path": str(model_path),
            "device": "cuda:0",
            "dtype": "bfloat16",
            "freeze_audio_encoder": True,
            "freeze_base_llm": True,
        },
        "lora": {
            "rank": 128,
            "alpha": 256,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "data": {
            "manifest_path": str(manifest),
            "audio_base_dir": str(root / "train"),
            "split_contract_path": str(split_contract),
            "data_gate_summary_path": str(data_summary),
            "human_audit_summary_path": str(human_summary),
            "require_human_audit": True,
            "expected_split": "train",
            "pre_split": True,
            "sample_rate": 16000,
            "max_audio_seconds": 20.0,
            "style": "detailed",
            "target_mode": "atomic",
            "include_lyrics": False,
            "slot_aware": True,
        },
        "train": {
            "steps": args.steps,
            "global_effective_batch": 8,
            "micro_batch_size": 1,
            "learning_rate": 5.0e-5,
            "schedule": "cosine",
            "warmup_steps": min(100, max(10, args.steps // 10)),
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "seed": 20260816,
            "output_dir": str(root / "model" / "b3_real_complex_anchor"),
            "shuffle_events": True,
        },
        "loss": {
            "text_weight": 1.0,
            "type_weight": 2.0,
            "timestamp_weight": 5.0,
        },
        "experiment_contract": {
            "dataset_id": data_gate["dataset_id"],
            "test_only_evaluation": True,
            "legacy_v6_comparison_is_diagnostic_only": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"config={output}")
    print(f"dataset_id={data_gate['dataset_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
