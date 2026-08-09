"""Tests that prevent low-coverage boundary metrics from selecting a model."""

from sceneledger.eval.selection import (
    coverage_aware_metrics,
    select_eventness_threshold,
)


def test_coverage_aware_metrics_weights_only_matched_boundary_errors() -> None:
    metrics = {
        "samples": [
            {"n_ref": 10, "n_hyp": 1, "n_matched": 1, "onset_mae": 0.01, "offset_mae": 0.02},
            {"n_ref": 10, "n_hyp": 0, "n_matched": 0, "onset_mae": 0.0, "offset_mae": 0.0},
        ]
    }
    result = coverage_aware_metrics(metrics)
    assert result["matched_onset_mae"] == 0.01
    assert result["matched_boundary_count"] == 1
    assert result["boundary_reference_coverage"] == 0.05
    assert result["micro_event_recall"] == 0.05


def test_threshold_selection_prefers_event_f1_over_tiny_boundary_mae() -> None:
    low_coverage = {
        "threshold": 0.4,
        "micro_event_f1": 0.02,
        "micro_event_recall": 0.01,
        "total_hallucination": 0,
        "matched_onset_mae": 0.008,
    }
    useful = {
        "threshold": 0.15,
        "micro_event_f1": 0.10,
        "micro_event_recall": 0.18,
        "total_hallucination": 30,
        "matched_onset_mae": 0.29,
    }
    assert select_eventness_threshold([low_coverage, useful]) is useful
