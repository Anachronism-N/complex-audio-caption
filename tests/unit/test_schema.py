"""Unit tests for the schema and its cross-field validators."""

from __future__ import annotations

import pytest
from fixtures.factory import ev, ledger, t, tr
from pydantic import ValidationError

from sceneledger.data.schema import Event, Span, Track


def test_span_quantizes_to_01_grid():
    s = Span(start_sec=0.123, end_sec=1.789)
    assert s.start_sec == 0.1
    assert s.end_sec == 1.8


def test_span_rejects_end_le_start():
    with pytest.raises(ValidationError):
        Span(start_sec=1.0, end_sec=1.0)
    with pytest.raises(ValidationError):
        Span(start_sec=2.0, end_sec=1.0)


def test_span_rejects_negative_start():
    with pytest.raises(ValidationError):
        Span(start_sec=-0.1, end_sec=1.0)


def test_track_rejects_overlapping_spans():
    with pytest.raises(ValidationError):
        tr(
            "T1",
            "speech",
            [t(0.0, 1.0), t(0.5, 1.5)],
        )


def test_event_rejects_overlapping_spans():
    with pytest.raises(ValidationError):
        ev("E1", "speech", [t(0.0, 1.0), t(0.7, 1.5)], text="hi")


def test_event_rejects_empty_text():
    with pytest.raises(ValidationError):
        ev("E1", "sfx", [t(0.0, 0.1)], text="")


def test_id_patterns_enforced():
    with pytest.raises(ValidationError):
        Track(id="X1", kind="speech", spans=[t(0, 1)], confidence=0.9)
    with pytest.raises(ValidationError):
        Event(id="X1", type="sfx", spans=[t(0, 1)], text="x", confidence=0.9)


def test_track_id_must_resolve():
    with pytest.raises(ValidationError):
        ledger(
            "s1",
            10.0,
            events=[ev("E1", "speech", [t(0, 1)], text="hi", track_id="T9")],
        )


def test_duplicate_event_ids_rejected():
    with pytest.raises(ValidationError):
        ledger(
            "s1",
            10.0,
            events=[
                ev("E1", "sfx", [t(0, 1)], text="a"),
                ev("E1", "sfx", [t(2, 3)], text="b"),
            ],
        )


def test_event_past_duration_rejected():
    with pytest.raises(ValidationError):
        ledger(
            "s1",
            5.0,
            events=[ev("E1", "sfx", [t(4.0, 6.0)], text="x")],
        )


def test_relation_target_must_exist():
    from sceneledger.data.schema import Relation

    with pytest.raises(ValidationError):
        ledger(
            "s1",
            10.0,
            events=[
                ev("E1", "sfx", [t(0, 1)], text="a"),
                ev(
                    "E2",
                    "sfx",
                    [t(2, 3)],
                    text="b",
                ).model_copy(
                    update={"relations": [Relation(predicate="before", target_event_id="E99")]}
                ),
            ],
        )


def test_multispan_event_allowed_when_disjoint():
    lg = ledger(
        "s1",
        10.0,
        events=[ev("E1", "sfx", [t(0.0, 0.5), t(2.0, 2.5)], text="glass twice")],
    )
    assert lg.events[0].start_sec() == 0.0
    assert lg.events[0].end_sec() == 2.5


def test_empty_scene_allowed():
    lg = ledger("empty", 10.0)
    assert lg.event_count() == 0
    assert lg.track_count() == 0
