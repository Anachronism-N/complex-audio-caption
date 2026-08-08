from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .matching import boundary_error, match_events
from .types import Ledger


@dataclass
class CorpusMetrics:
    samples: int
    reference_events: int
    predicted_events: int
    matched_events: int
    event_precision: float
    event_recall: float
    event_f1: float
    mean_temporal_iou: float
    mean_semantic_f1: float
    onset_mae_sec: float
    offset_mae_sec: float
    track_pointer_accuracy: float
    source_count_mae: float
    per_type: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_corpus(references: Iterable[Ledger], predictions: Iterable[Ledger]) -> CorpusMetrics:
    reference_by_id = {ledger.sample_id: ledger for ledger in references}
    prediction_by_id = {ledger.sample_id: ledger for ledger in predictions}
    missing = sorted(set(reference_by_id) - set(prediction_by_id))
    extra = sorted(set(prediction_by_id) - set(reference_by_id))
    if missing or extra:
        raise ValueError(f"Sample ID mismatch. missing={missing[:5]}, extra={extra[:5]}")

    total_ref = total_pred = 0
    all_matches = []
    onset_errors: list[float] = []
    offset_errors: list[float] = []
    count_errors: list[float] = []
    type_counts = defaultdict(lambda: {"reference": 0, "prediction": 0, "matched": 0})

    for sample_id, reference in reference_by_id.items():
        prediction = prediction_by_id[sample_id]
        matches = match_events(reference.events, prediction.events)
        total_ref += len(reference.events)
        total_pred += len(prediction.events)
        all_matches.extend(matches)
        count_errors.append(abs(len(reference.tracks) - len(prediction.tracks)))
        for event in reference.events:
            type_counts[event.type]["reference"] += 1
        for event in prediction.events:
            type_counts[event.type]["prediction"] += 1
        for match in matches:
            event_type = reference.events[match.reference_index].type
            type_counts[event_type]["matched"] += 1
            onset, offset = boundary_error(
                reference.events[match.reference_index], prediction.events[match.prediction_index]
            )
            onset_errors.append(onset)
            offset_errors.append(offset)

    matched = len(all_matches)
    precision = matched / total_pred if total_pred else (1.0 if total_ref == 0 else 0.0)
    recall = matched / total_ref if total_ref else (1.0 if total_pred == 0 else 0.0)
    f1 = _f1(precision, recall)
    per_type: dict[str, dict[str, float]] = {}
    for event_type, counts in sorted(type_counts.items()):
        type_precision = counts["matched"] / counts["prediction"] if counts["prediction"] else 0.0
        type_recall = counts["matched"] / counts["reference"] if counts["reference"] else 0.0
        per_type[event_type] = {
            **{key: float(value) for key, value in counts.items()},
            "precision": type_precision,
            "recall": type_recall,
            "f1": _f1(type_precision, type_recall),
        }

    return CorpusMetrics(
        samples=len(reference_by_id),
        reference_events=total_ref,
        predicted_events=total_pred,
        matched_events=matched,
        event_precision=precision,
        event_recall=recall,
        event_f1=f1,
        mean_temporal_iou=_mean([match.temporal_iou for match in all_matches]),
        mean_semantic_f1=_mean([match.semantic_f1 for match in all_matches]),
        onset_mae_sec=_mean(onset_errors),
        offset_mae_sec=_mean(offset_errors),
        track_pointer_accuracy=_mean([float(match.track_correct) for match in all_matches]),
        source_count_mae=_mean(count_errors),
        per_type=per_type,
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
