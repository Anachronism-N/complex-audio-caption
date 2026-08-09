"""Tests for strict S1 experiment evidence collection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.collect_s1_results import collect


def _write_run(root: Path, *, leakage: int = 0) -> None:
    model = root / "main/model"
    model.mkdir(parents=True)
    metrics = {
        "macro_event_f1": 0.4,
        "macro_event_precision": 0.5,
        "macro_event_recall": 0.3,
        "micro_event_f1": 0.38,
        "micro_event_precision": 0.45,
        "micro_event_recall": 0.33,
        "macro_seg_f1_100ms": 0.35,
        "mean_onset_mae": 0.2,
        "mean_offset_mae": 0.3,
        "total_hallucination": 4,
        "total_omission": 5,
        "matched_boundary_count": 7,
        "boundary_reference_coverage": 0.33,
        "matched_onset_mae": 0.21,
        "matched_offset_mae": 0.31,
    }
    summary = {
        "git_commit": "abc",
        "config_sha256": "cfg",
        "n_validation": 10,
        "eventness_threshold": 0.5,
        "activity_threshold": 0.5,
        "tiou_threshold": 0.3,
        "metrics": metrics,
    }
    (model / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (model / "run_manifest.json").write_text(
        json.dumps({"config_sha256": "cfg"}), encoding="utf-8"
    )
    (model / "split.json").write_text(
        json.dumps({"source_leakage_count": leakage}), encoding="utf-8"
    )


def test_collect_s1_results_checks_evidence_and_flattens_metrics(tmp_path: Path) -> None:
    _write_run(tmp_path)
    rows = collect(tmp_path)
    assert rows[0]["run"] == "main"
    assert rows[0]["macro_event_f1"] == 0.4


def test_collect_s1_results_rejects_leakage(tmp_path: Path) -> None:
    _write_run(tmp_path, leakage=1)
    with pytest.raises(ValueError, match="source leakage"):
        collect(tmp_path)
