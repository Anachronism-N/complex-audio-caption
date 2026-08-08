"""Unit tests for the tolerant model-output parser."""

from __future__ import annotations

from sceneledger.eval.parser import parse_model_output
from fixtures.factory import t


def _xml(text: str, sample_id: str = "s1", duration: float = 30.0):
    ledger, report = parse_model_output(text, sample_id=sample_id, duration_sec=duration)
    return ledger, report


def test_strict_parse_path_sets_flag():
    good = (
        '<ledger schema_version="0.2.0" sample_id="s1" duration="10" time_resolution="0.1">'
        '<events><sfx id="E1" t="0-0.5" confidence="0.9">glass</sfx></events></ledger>'
    )
    lg, rep = _xml(good)
    assert rep.strict_format_success is True
    assert rep.ok is True
    assert len(lg.events) == 1


def test_recovers_from_surrounding_prose():
    text = (
        "Here is the caption:\n"
        '<speech id="E1" track="T1" t="0.7-2.9" confidence="0.9">hello world</speech>'
        " and a music track:"
        '<music id="E2" track="T2" t="0-10" confidence="0.8">rock</music>'
    )
    lg, rep = _xml(text, duration=10.0)
    assert rep.strict_format_success is False
    assert len(lg.events) == 2
    assert {e.type for e in lg.events} == {"speech", "music"}
    assert rep.events_recovered == 2


def test_default_confidence_when_missing():
    text = '<sfx id="E1" t="0-0.5">glass</sfx>'
    lg, rep = _xml(text)
    assert lg.events[0].confidence == 1.0
    assert rep.strict_format_success is False
    assert any("confidence" in w for w in rep.warnings)


def test_infers_tracks_from_events():
    text = (
        '<speech id="E1" track="T1" t="0.7-2.9" confidence="0.9">hi</speech>'
        '<speech id="E2" track="T1" t="3.0-5.0" confidence="0.9">bye</speech>'
        '<music id="E3" track="T2" t="0-5" confidence="0.8">rock</music>'
    )
    lg, rep = _xml(text, duration=5.0)
    track_ids = {t.id for t in lg.tracks}
    assert track_ids == {"T1", "T2"}
    t1 = next(t for t in lg.tracks if t.id == "T1")
    assert t1.kind == "speech"
    # 0.1 s gap between [0.7,2.9] and [3.0,5.0] -> NOT merged
    assert len(t1.spans) == 2
    assert t1.spans[0].start_sec == 0.7
    assert t1.spans[0].end_sec == 2.9
    assert t1.spans[1].end_sec == 5.0


def test_rejects_event_with_missing_time():
    text = '<sfx id="E1" confidence="0.9">glass</sfx>'
    lg, rep = _xml(text)
    assert len(lg.events) == 0
    assert rep.events_rejected == 1


def test_rejects_event_with_empty_text():
    text = '<sfx id="E1" t="0-0.5" confidence="0.9">   </sfx>'
    lg, rep = _xml(text)
    assert rep.events_rejected == 1
    assert len(lg.events) == 0


def test_malformed_track_ref_dropped():
    text = '<speech id="E1" track="speaker-A" t="0-1" confidence="0.9">hi</speech>'
    lg, rep = _xml(text, duration=1.0)
    assert lg.events[0].track_id is None


def test_multispan_parse():
    text = '<sfx id="E1" t="0.4-0.8,1.6-2.0" confidence="0.9">double</sfx>'
    lg, rep = _xml(text, duration=2.0)
    assert len(lg.events[0].spans) == 2


def test_smart_quotes_and_case_insensitive_tags():
    text = '<SPEECH id="E1" t="0-1" confidence="0.9">“hello”</SPEECH>'
    lg, rep = _xml(text, duration=1.0)
    assert len(lg.events) == 1
    assert lg.events[0].type == "speech"
