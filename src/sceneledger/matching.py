from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Event, Span


@dataclass(frozen=True)
class MatchWeights:
    temporal: float = 0.55
    semantic: float = 0.35
    track: float = 0.10


@dataclass(frozen=True)
class EventMatch:
    reference_index: int
    prediction_index: int
    temporal_iou: float
    semantic_f1: float
    track_correct: bool
    score: float


DEFAULT_MATCH_WEIGHTS = MatchWeights()


def match_events(
    reference: list[Event],
    prediction: list[Event],
    *,
    weights: MatchWeights = DEFAULT_MATCH_WEIGHTS,
    minimum_score: float = 0.25,
) -> list[EventMatch]:
    if not reference or not prediction:
        return []
    scores = np.full((len(reference), len(prediction)), -1.0, dtype=np.float64)
    details: dict[tuple[int, int], tuple[float, float, bool]] = {}
    for i, ref in enumerate(reference):
        for j, pred in enumerate(prediction):
            if ref.type != pred.type:
                continue
            temporal = temporal_iou(ref.spans, pred.spans)
            semantic = token_f1(ref.text, pred.text)
            track_correct = ref.track_id is None or ref.track_id == pred.track_id
            track_score = 1.0 if track_correct else 0.0
            score = (
                weights.temporal * temporal
                + weights.semantic * semantic
                + weights.track * track_score
            )
            scores[i, j] = score
            details[(i, j)] = (temporal, semantic, track_correct)

    rows, columns = linear_sum_assignment(-scores)
    matches: list[EventMatch] = []
    for i, j in zip(rows.tolist(), columns.tolist()):
        score = float(scores[i, j])
        if score < minimum_score:
            continue
        temporal, semantic, track_correct = details[(i, j)]
        matches.append(EventMatch(i, j, temporal, semantic, track_correct, score))
    return matches


def temporal_iou(left: list[Span], right: list[Span]) -> float:
    left_intervals = _merge(left)
    right_intervals = _merge(right)
    intersection = 0.0
    i = j = 0
    while i < len(left_intervals) and j < len(right_intervals):
        left_start, left_end = left_intervals[i]
        right_start, right_end = right_intervals[j]
        intersection += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            i += 1
        else:
            j += 1
    left_duration = sum(end - start for start, end in left_intervals)
    right_duration = sum(end - start for start, end in right_intervals)
    union = left_duration + right_duration - intersection
    return intersection / union if union > 0 else 0.0


def boundary_error(reference: Event, prediction: Event) -> tuple[float, float]:
    return (
        abs(
            min(span.start_sec for span in reference.spans)
            - min(span.start_sec for span in prediction.spans)
        ),
        abs(
            max(span.end_sec for span in reference.spans)
            - max(span.end_sec for span in prediction.spans)
        ),
    )


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = _tokens(reference)
    pred_tokens = _tokens(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = sum(min(count, pred_counts.get(token, 0)) for token, count in ref_counts.items())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _tokens(text: str) -> list[str]:
    normalized = text.lower().strip()
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", normalized)


def _merge(spans: list[Span]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for span in sorted(spans):
        if result and span.start_sec <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], span.end_sec))
        else:
            result.append((span.start_sec, span.end_sec))
    return result
