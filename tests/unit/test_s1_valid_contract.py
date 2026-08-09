"""CPU-only contract checks for the server-side S1a-valid experiment."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_s1_config_uses_explicit_leakage_safe_folds() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/model/s1_event_slots.yaml").read_text(encoding="utf-8")
    )
    data = config["data"]
    assert data["train_manifest_path"].endswith("sft/train_manifest.jsonl")
    assert data["val_manifest_path"].endswith("sft/val_manifest.jsonl")
    assert data["group_key"] == "source_id"
    assert config["train"]["deterministic"] is True
    assert config["evaluation"]["eventness_threshold"] == 0.5


def test_s1_runner_fails_closed_and_supports_resumption() -> None:
    script = (ROOT / "scripts/run_s1_valid.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "run scripts/run_b3_valid.sh first" in script
    assert "--train-manifest" in script
    assert "--val-manifest" in script
    assert "--resume" in script
    assert "--evaluate-checkpoint" in script


def test_downstream_server_runners_enforce_anchor_gate() -> None:
    for name in ("run_b1_official.sh", "run_b3_valid.sh", "run_s1_valid.sh"):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "require_anchor_pass.py" in script
        assert "TAG_SUMMARY" in script
