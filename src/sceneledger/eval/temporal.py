"""Temporal metrics on event spans.

All metrics operate on the 0.1 s grid. Events may have multiple disjoint
spans; ``multi_span_iou`` treats the union of spans as the event's active set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sceneledger.data.schema import TIME_RESOLUTION_SEC, Event, Span


def _span_union_seconds(spans: list[Span]) -> list[tuple[float, float]]:
    """Merge overlapping spans; return sorted (start, end) tuples."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s.start_sec)
    merged: list[tuple[float, float]] = [(spans[0].start_sec, spans[0].end_sec)]
    for cur in spans[1:]:
        last_start, last_end = merged[-1]
        if cur.start_sec <= last_end - 1e-9:
            merged[-1] = (last_start, max(last_end, cur.end_sec))
        else:
            merged.append((cur.start_sec, cur.end_sec))
    return merged


def _union_duration(spans: list[Span]) -> float:
    return round(sum(e - s for s, e in _span_union_seconds(spans)), 6)


def _intersection_duration(a: list[Span], b: list[Span]) -> float:
    """Length of the intersection of two span unions (merged intervals)."""
    ua = _span_union_seconds(a)
    ub = _span_union_seconds(b)
    i = j = 0
    total = 0.0
    while i < len(ua) and j < len(ub):
        s = max(ua[i][0], ub[j][0])
        e = min(ua[i][1], ub[j][1])
        if e > s:
            total += e - s
        if ua[i][1] < ub[j][1]:
            i += 1
        else:
            j += 1
    return round(total, 6)


def multi_span_iou(a: list[Span], b: list[Span]) -> float:
    """IoU over the union of all spans of two events."""
    inter = _intersection_duration(a, b)
    union = _union_duration(a) + _union_duration(b) - inter
    if union <= 0:
        return 0.0
    return round(inter / union, 6)


def temporal_tiou(a: Event, b: Event) -> float:
    """tIoU = intersection-over-union of two events' span unions."""
    return multi_span_iou(a.spans, b.spans)


@dataclass
class BoundaryErrors:
    onset_mae: float
    offset_mae: float
    onset_p90: float
    offset_p90: float


def boundary_mae(
    matches: list[tuple[Event, Event]],
) -> BoundaryErrors:
    """Onset/offset MAE and p90 absolute error (seconds) over matched pairs.

    Onset = earliest span start; offset = latest span end.
    """
    if not matches:
        return BoundaryErrors(0.0, 0.0, 0.0, 0.0)
    onsets = np.array(
        [abs(ref.start_sec() - hyp.start_sec()) for ref, hyp in matches]
    )
    offsets = np.array(
        [abs(ref.end_sec() - hyp.end_sec()) for ref, hyp in matches]
    )
    return BoundaryErrors(
        onset_mae=round(float(onsets.mean()), 6),
        offset_mae=round(float(offsets.mean()), 6),
        onset_p90=round(float(np.percentile(onsets, 90)), 6),
        offset_p90=round(float(np.percentile(offsets, 90)), 6),
    )


def tolerance_accuracy(
    matches: list[tuple[Event, Event]],
    collar_seconds: float = 0.5,
) -> float:
    """Fraction of matched pairs whose onset AND offset are within ``collar``.

    Vacuously 1.0 when there are no matched pairs (no boundary violations).
    """
    if not matches:
        return 1.0
    ok = sum(
        1
        for ref, hyp in matches
        if abs(ref.start_sec() - hyp.start_sec()) <= collar_seconds + 1e-9
        and abs(ref.end_sec() - hyp.end_sec()) <= collar_seconds + 1e-9
    )
    return round(ok / len(matches), 6)


def seg_f1(
    refs: list[Event],
    hyps: list[Event],
    collar_seconds: float = 0.1,
) -> tuple[float, float, float]:
    """Frame-level segment F1 (AudioSet STRONG / TAC style).

    Builds per-type binary activity at the 0.1 s grid from each side's span
    unions, then counts frame-level TP/FP/FN. Returns (precision, recall, f1).
    ``collar_seconds`` is accepted for API compatibility but frame-level F1
    uses direct overlap; use :func:`tolerance_accuracy` for collar-based
    boundary acceptance.
    """
    grid = TIME_RESOLUTION_SEC
    if not refs and not hyps:
        return 1.0, 1.0, 1.0
    T = max([e.end_sec() for e in refs + hyps] + [0.0])
    n_frames = int(round(T / grid)) + 1

    types = sorted({e.type for e in refs + hyps})
    tp = fp = fn = 0
    for t in types:
        ref_arr = np.zeros(n_frames)
        hyp_arr = np.zeros(n_frames)
        for ev in refs:
            if ev.type != t:
                continue
            for s, e in _span_union_seconds(ev.spans):
                for fr in range(int(round(s / grid)), int(round(e / grid))):
                    if 0 <= fr < n_frames:
                        ref_arr[fr] = 1.0
        for ev in hyps:
            if ev.type != t:
                continue
            for s, e in _span_union_seconds(ev.spans):
                for fr in range(int(round(s / grid)), int(round(e / grid))):
                    if 0 <= fr < n_frames:
                        hyp_arr[fr] = 1.0
        tp += int(np.sum((ref_arr > 0) & (hyp_arr > 0)))
        fp += int(np.sum((ref_arr == 0) & (hyp_arr > 0)))
        fn += int(np.sum((ref_arr > 0) & (hyp_arr == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 6), round(recall, 6), round(f1, 6)


__all__ = [
    "BoundaryErrors",
    "boundary_mae",
    "multi_span_iou",
    "seg_f1",
    "temporal_tiou",
    "tolerance_accuracy",
]
