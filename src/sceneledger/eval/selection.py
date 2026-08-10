"""Coverage-aware result summaries and calibration-only threshold selection."""

from __future__ import annotations


def coverage_aware_metrics(metrics: dict) -> dict[str, float | int]:
    """Pair matched-event boundary error with reference coverage."""
    samples = metrics.get("samples", [])
    n_reference = sum(int(sample["n_ref"]) for sample in samples)
    n_hypothesis = sum(int(sample["n_hyp"]) for sample in samples)
    n_matched = sum(int(sample["n_matched"]) for sample in samples)
    precision = n_matched / n_hypothesis if n_hypothesis else 1.0
    recall = n_matched / n_reference if n_reference else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if n_matched:
        onset_mae = sum(
            float(sample["onset_mae"]) * int(sample["n_matched"])
            for sample in samples
        ) / n_matched
        offset_mae = sum(
            float(sample["offset_mae"]) * int(sample["n_matched"])
            for sample in samples
        ) / n_matched
    else:
        onset_mae = 0.0
        offset_mae = 0.0
    return {
        "micro_event_precision": round(precision, 6),
        "micro_event_recall": round(recall, 6),
        "micro_event_f1": round(f1, 6),
        "matched_boundary_count": n_matched,
        "boundary_reference_coverage": round(
            n_matched / n_reference if n_reference else 1.0, 6
        ),
        "matched_onset_mae": round(onset_mae, 6),
        "matched_offset_mae": round(offset_mae, 6),
    }


def select_eventness_threshold(rows: list[dict]) -> dict:
    """Select eventness by detection quality, not conditional boundary MAE."""
    if not rows:
        raise ValueError("threshold calibration produced no candidates")
    return max(
        rows,
        key=lambda row: (
            row["micro_event_f1"],
            row["micro_event_recall"],
            -row["total_hallucination"],
            row["threshold"],
        ),
    )


__all__ = ["coverage_aware_metrics", "select_eventness_threshold"]
