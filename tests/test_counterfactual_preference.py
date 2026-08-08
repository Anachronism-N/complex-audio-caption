from __future__ import annotations

import json
from pathlib import Path

from sceneledger.counterfactual import evaluate_carc
from sceneledger.preference import build_preference_rows
from sceneledger.types import Event, Ledger, Span, Track


def _ledger(sample_id: str, *, delta_start: float | None = None) -> Ledger:
    tracks = [Track(id="T1", kind="music", spans=[Span(0.0, 3.0)], confidence=1.0)]
    events = [
        Event(
            id="E1",
            type="music",
            track_id="T1",
            spans=[Span(0.0, 3.0)],
            text="steady music",
            confidence=1.0,
        )
    ]
    if delta_start is not None:
        tracks.append(
            Track(
                id="T2",
                kind="sfx",
                spans=[Span(delta_start, delta_start + 0.5)],
                confidence=1.0,
            )
        )
        events.append(
            Event(
                id="E2",
                type="sfx",
                track_id="T2",
                spans=[Span(delta_start, delta_start + 0.5)],
                text="a beep",
                confidence=1.0,
            )
        )
    return Ledger(sample_id=sample_id, duration_sec=3.0, tracks=tracks, events=events)


def _delta(start: float) -> dict:
    return {
        "id": "E_delta",
        "type": "sfx",
        "track_id": "T_delta",
        "spans": [{"start_sec": start, "end_sec": start + 0.5}],
        "text": "a beep",
        "confidence": 1.0,
    }


def test_counterfactual_metrics_separate_sensitivity_and_hallucination(tmp_path: Path) -> None:
    rows = [
        {
            "pair_id": "audible",
            "audibility_target": "must_add",
            "shift_delta_sec": 1.0,
            "delta_event": _delta(0.5),
            "shifted_delta_event": _delta(1.5),
        },
        {
            "pair_id": "hidden",
            "audibility_target": "must_not_add",
            "shift_delta_sec": 0.5,
            "delta_event": _delta(0.5),
            "shifted_delta_event": _delta(1.0),
        },
    ]
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    predictions = [
        _ledger("audible:before"),
        _ledger("audible:after", delta_start=0.5),
        _ledger("audible:shifted_after", delta_start=1.5),
        _ledger("hidden:before"),
        _ledger("hidden:after"),
        _ledger("hidden:shifted_after"),
    ]
    metrics = evaluate_carc(manifest, predictions)
    assert metrics.add_recall == 1.0
    assert metrics.removal_success == 1.0
    assert metrics.hidden_addition_rate == 0.0
    assert metrics.shift_equivariance_mae_sec == 0.0
    assert metrics.background_event_preservation_recall == 1.0


def test_preference_negatives_are_deterministic_and_different(tmp_path: Path) -> None:
    ledger = _ledger("sample", delta_start=0.5)
    audio_paths = {"sample": tmp_path / "sample.wav"}
    first = build_preference_rows(
        [ledger], audio_paths, negatives_per_sample=7, seed=9
    )
    second = build_preference_rows(
        [ledger], audio_paths, negatives_per_sample=7, seed=9
    )
    assert first == second
    assert len(first) == 7
    assert len({row["negative_type"] for row in first}) >= 5
    assert all(row["chosen"] != row["rejected"] for row in first)
