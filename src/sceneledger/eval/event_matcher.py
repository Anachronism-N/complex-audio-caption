"""Permutation-invariant event matching via Hungarian assignment.

Reference and hypothesis events are both unordered sets; to compare them we
must first decide which hypothesis corresponds to which reference. We solve a
balanced assignment problem with a cost that combines:

* type agreement (hard gate: different types cannot match),
* temporal IoU over span unions,
* text similarity (token F1),
Track IDs are intentionally excluded from event assignment by default because
``T1``/``T2`` labels are permutation arbitrary. Pointer quality is evaluated
after event matching with a separate optimal track-label alignment.

``scipy.optimize.linear_sum_assignment`` gives the optimal one-to-one matching.
Unmatched references are *omissions*; unmatched hypotheses are *hallucinations*
(or false alarms).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from sceneledger.data.schema import Event
from sceneledger.eval.temporal import temporal_tiou

TextSim = Callable[[str, str], float]


@dataclass
class EventMatch:
    ref_id: str | None
    hyp_id: str | None
    type: str | None
    tiou: float
    text_sim: float
    track_match: bool
    is_match: bool  # True iff a valid one-to-one match passed the threshold


def token_f1(a: str, b: str) -> float:
    """Char-normalized token overlap F1 (whitespace tokens, lowercase)."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    prec = inter / len(tb)
    rec = inter / len(ta)
    return round(2 * prec * rec / (prec + rec), 6)


def _track_agreement(ref: Event, hyp: Event) -> bool:
    if ref.track_id is None and hyp.track_id is None:
        return True
    if ref.track_id is None or hyp.track_id is None:
        return False
    return ref.track_id == hyp.track_id


def match_events(
    refs: list[Event],
    hyps: list[Event],
    *,
    tiou_threshold: float = 0.3,
    text_sim: TextSim = token_f1,
    text_weight: float = 0.3,
    track_weight: float = 0.0,
) -> list[EventMatch]:
    """Return the optimal one-to-one matching between ``refs`` and ``hyps``.

    A pair is a valid match iff same type AND ``tIoU >= tiou_threshold``.
    The assignment cost is minimized over valid pairs; invalid pairs have
    infinite cost. The result is permutation-invariant: reordering ``hyps``
    does not change the set of matches.
    """
    n, m = len(refs), len(hyps)
    INF = 1e6
    size = max(n, m)
    cost = np.full((size, size), INF, dtype=float)
    pair_info: dict[tuple[int, int], tuple[float, float, bool]] = {}

    for i, ref in enumerate(refs):
        for j, hyp in enumerate(hyps):
            if ref.type != hyp.type:
                continue
            tiou = temporal_tiou(ref, hyp)
            if tiou < tiou_threshold:
                continue
            tsim = text_sim(ref.text, hyp.text)
            track_ok = _track_agreement(ref, hyp)
            # cost in [0, 1]; lower is better
            c = (
                (1.0 - tiou)
                + text_weight * (1.0 - tsim)
                + track_weight * (0.0 if track_ok else 1.0)
            )
            cost[i, j] = c
            pair_info[(i, j)] = (tiou, tsim, track_ok)

    if size == 0:
        return []

    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[EventMatch] = []
    matched_ref: set[int] = set()
    matched_hyp: set[int] = set()
    for i, j in zip(row_ind, col_ind, strict=True):
        if i >= n or j >= m:
            continue  # padding row/col
        if cost[i, j] >= INF:
            continue
        tiou, tsim, track_ok = pair_info[(i, j)]
        matches.append(
            EventMatch(
                ref_id=refs[i].id,
                hyp_id=hyps[j].id,
                type=refs[i].type,
                tiou=tiou,
                text_sim=tsim,
                track_match=track_ok,
                is_match=True,
            )
        )
        matched_ref.add(i)
        matched_hyp.add(j)

    for i, ref in enumerate(refs):
        if i not in matched_ref:
            matches.append(
                EventMatch(
                    ref_id=ref.id,
                    hyp_id=None,
                    type=ref.type,
                    tiou=0.0,
                    text_sim=0.0,
                    track_match=False,
                    is_match=False,
                )
            )
    for j, hyp in enumerate(hyps):
        if j not in matched_hyp:
            matches.append(
                EventMatch(
                    ref_id=None,
                    hyp_id=hyp.id,
                    type=hyp.type,
                    tiou=0.0,
                    text_sim=0.0,
                    track_match=False,
                    is_match=False,
                )
            )
    return matches


def permutation_invariant_pointer_accuracy(
    matches: list[EventMatch], refs: list[Event], hyps: list[Event]
) -> float:
    """Score event-to-track grouping after optimal track-label alignment.

    The score cannot compare raw ``T1`` labels: exchanging all hypothesis
    labels must leave the result unchanged.  We build a reference-track by
    hypothesis-track contingency table over matched events and use Hungarian
    assignment to find the best one-to-one renaming. Events without a pointer
    remain in the denominator and therefore count as incorrect.
    """
    pairs = matched_pairs(matches, refs, hyps)
    if not pairs:
        return 1.0 if not refs and not hyps else 0.0
    ref_tracks = sorted(
        {ref.track_id for ref, _hyp in pairs if ref.track_id is not None}
    )
    hyp_tracks = sorted(
        {hyp.track_id for _ref, hyp in pairs if hyp.track_id is not None}
    )
    if not ref_tracks or not hyp_tracks:
        return 0.0
    ref_index = {track_id: index for index, track_id in enumerate(ref_tracks)}
    hyp_index = {track_id: index for index, track_id in enumerate(hyp_tracks)}
    counts = np.zeros((len(ref_tracks), len(hyp_tracks)), dtype=np.int64)
    for ref, hyp in pairs:
        if ref.track_id is None or hyp.track_id is None:
            continue
        counts[ref_index[ref.track_id], hyp_index[hyp.track_id]] += 1
    row_ind, col_ind = linear_sum_assignment(-counts)
    correct = int(counts[row_ind, col_ind].sum())
    # Unmatched reference events are omitted pointers, not free exclusions.
    # Hallucinated hypothesis events are reported separately by the event
    # metrics, while every reference event remains in this denominator.
    return correct / len(refs)


def matched_pairs(
    matches: list[EventMatch], refs: list[Event], hyps: list[Event]
) -> list[tuple[Event, Event]]:
    """Project :class:`EventMatch` back to (ref, hyp) event objects for temporal metrics."""
    ref_by_id = {e.id: e for e in refs}
    hyp_by_id = {e.id: e for e in hyps}
    pairs: list[tuple[Event, Event]] = []
    for mm in matches:
        if not mm.is_match:
            continue
        if mm.ref_id in ref_by_id and mm.hyp_id in hyp_by_id:
            pairs.append((ref_by_id[mm.ref_id], hyp_by_id[mm.hyp_id]))
    return pairs


__all__ = [
    "EventMatch",
    "TextSim",
    "match_events",
    "matched_pairs",
    "permutation_invariant_pointer_accuracy",
    "token_f1",
]
