from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .matching import boundary_error, match_events, temporal_iou, token_f1
from .types import Event, Ledger


@dataclass
class CounterfactualMetrics:
    pairs: int
    must_add_pairs: int
    must_not_add_pairs: int
    add_recall: float
    removal_success: float
    pre_intervention_hallucination_rate: float
    hidden_addition_rate: float
    shift_detection_recall: float
    shift_equivariance_mae_sec: float
    background_event_preservation_recall: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_carc(
    pair_manifest: str | Path, predictions: list[Ledger]
) -> CounterfactualMetrics:
    """Evaluate add/remove/shift consistency on Exact-CARC predictions.

    Prediction IDs must be ``{pair_id}:before``, ``{pair_id}:after`` and
    ``{pair_id}:shifted_after``. Hidden interventions test hallucination
    suppression; audible interventions test sensitivity and equivariance.
    """
    rows = [
        json.loads(line)
        for line in Path(pair_manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {ledger.sample_id: ledger for ledger in predictions}
    add_hits: list[float] = []
    removal_hits: list[float] = []
    before_hallucinations: list[float] = []
    hidden_additions: list[float] = []
    shift_hits: list[float] = []
    shift_errors: list[float] = []
    preservation: list[float] = []

    for row in rows:
        pair_id = row["pair_id"]
        required = {
            name: f"{pair_id}:{name}" for name in ("before", "after", "shifted_after")
        }
        missing = [sample_id for sample_id in required.values() if sample_id not in by_id]
        if missing:
            raise ValueError(f"Missing CARC predictions: {missing}")
        before = by_id[required["before"]]
        after = by_id[required["after"]]
        shifted = by_id[required["shifted_after"]]
        delta = Event.from_dict(row["delta_event"])
        shifted_delta = Event.from_dict(row["shifted_delta_event"])

        before_hit = _semantic_delta_index(delta, before.events)
        after_hit = _grounded_delta_index(delta, after.events)
        shifted_hit = _grounded_delta_index(shifted_delta, shifted.events)
        audibility = row["audibility_target"]
        if audibility == "must_add":
            add_hits.append(float(after_hit is not None))
            removal_hits.append(float(after_hit is not None and before_hit is None))
            before_hallucinations.append(float(before_hit is not None))
            shift_hits.append(float(shifted_hit is not None))
            if after_hit is not None and shifted_hit is not None:
                predicted_shift = (
                    shifted.events[shifted_hit].onset - after.events[after_hit].onset
                )
                shift_errors.append(abs(predicted_shift - float(row["shift_delta_sec"])))
        elif audibility == "must_not_add":
            hidden_additions.append(float(after_hit is not None))

        after_background = [
            event for index, event in enumerate(after.events) if index != after_hit
        ]
        matches = match_events(before.events, after_background)
        preservation.append(len(matches) / len(before.events) if before.events else 1.0)

    return CounterfactualMetrics(
        pairs=len(rows),
        must_add_pairs=len(add_hits),
        must_not_add_pairs=len(hidden_additions),
        add_recall=_mean(add_hits),
        removal_success=_mean(removal_hits),
        pre_intervention_hallucination_rate=_mean(before_hallucinations),
        hidden_addition_rate=_mean(hidden_additions),
        shift_detection_recall=_mean(shift_hits),
        shift_equivariance_mae_sec=_mean(shift_errors),
        background_event_preservation_recall=_mean(preservation),
    )


def _semantic_delta_index(delta: Event, events: list[Event]) -> int | None:
    candidates = [
        (token_f1(delta.text, event.text), index)
        for index, event in enumerate(events)
        if event.type == delta.type
    ]
    if not candidates:
        return None
    score, index = max(candidates)
    return index if score >= 0.5 else None


def _grounded_delta_index(delta: Event, events: list[Event]) -> int | None:
    candidates = []
    for index, event in enumerate(events):
        if event.type != delta.type:
            continue
        semantic = token_f1(delta.text, event.text)
        temporal = temporal_iou(delta.spans, event.spans)
        if semantic >= 0.5 and temporal >= 0.1:
            onset, offset = boundary_error(delta, event)
            candidates.append((0.7 * temporal + 0.3 * semantic - 0.01 * (onset + offset), index))
    return max(candidates)[1] if candidates else None


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0
