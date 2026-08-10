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
    assert config["train"]["calibration_fraction"] == 0.1
    assert config["evaluation"]["eventness_threshold"] == 0.5
    assert config["evaluation"]["primary_decode_mode"] == "hybrid"
    assert config["evaluation"]["decode_modes"] == [
        "activity",
        "boundary",
        "hybrid",
    ]
    assert len(config["evaluation"]["calibration_thresholds"]) > 1
    assert config["loss"]["activity_cost_weight"] > 0
    assert config["loss"]["boundary_cost_weight"] > 0


def test_s1_runner_fails_closed_and_supports_resumption() -> None:
    script = (ROOT / "scripts/run_s1_valid.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "run scripts/run_b3_valid.sh first" in script
    assert "--train-manifest" in script
    assert "--val-manifest" in script
    assert "--resume" in script
    assert "--evaluate-checkpoint" in script
    assert "activity_only" not in script  # ablations belong to the matrix runner


def test_downstream_server_runners_enforce_anchor_gate() -> None:
    for name in ("run_b1_official.sh", "run_b3_valid.sh", "run_s1_valid.sh"):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "require_anchor_pass.py" in script
        assert "TAG_SUMMARY" in script


def test_b3_data_runner_is_staged_and_persists_acceptance_summary() -> None:
    script = (ROOT / "scripts/run_b3_data.sh").read_text(encoding="utf-8")
    for stage in ("sources", "render", "export", "audit", "all"):
        assert stage in script
    assert "validation_report.json" in script
    assert "data_reproduction_summary.json" in script
    assert "sceneledger.cli.validate_b3_data" in script


def test_model_runners_require_passed_b3_data() -> None:
    for name in ("run_b3_valid.sh", "run_s1_valid.sh"):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "require_b3_data_pass.py" in script
        assert "data_reproduction_summary.json" in script
        assert "B3_DATASET_ID" in script
    b3_script = (ROOT / "scripts/run_b3_valid.sh").read_text(encoding="utf-8")
    assert "--train-manifest" in b3_script
    assert 'sft/train_manifest.jsonl' in b3_script


def test_single_head_ablations_disable_loss_and_matching_cost() -> None:
    script = (ROOT / "scripts/run_s1_ablation.sh").read_text(encoding="utf-8")
    assert "activity_only --boundary-weight 0 --boundary-cost-weight 0" in script
    assert "boundary_only --activity-weight 0 --activity-cost-weight 0" in script
