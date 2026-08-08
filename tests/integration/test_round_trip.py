"""Integration: end-to-end metrics + 20 fixtures covering edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sceneledger.eval.metrics import evaluate_corpus, evaluate_sample
from sceneledger.models.serializer import serialize
from fixtures.factory import ev, ledger, t, tr


def _build_fixture_set():
    """Build 20 reference ledgers covering overlap/lyrics/echo/multispan/empty/edge."""
    out = []
    # 1: speech+music overlap
    out.append(
        ledger("f01", 10.0, events=[
            ev("E1", "speech", [t(0.5, 2.5)], text="hello there", track_id="T1"),
            ev("E2", "music", [t(0.0, 10.0)], text="rock", track_id="T2"),
        ], tracks=[tr("T1", "speech", [t(0.5, 2.5)]), tr("T2", "music", [t(0, 10)])])
    )
    # 2: lyrics + accompaniment
    out.append(
        ledger("f02", 8.0, events=[
            ev("E1", "lys", [t(1.0, 3.0)], text="take me home", track_id="T1", verbatim=True),
            ev("E2", "music", [t(0.0, 8.0)], text="piano ballad", track_id="T2"),
        ], tracks=[tr("T1", "vocal", [t(1, 3)]), tr("T2", "music", [t(0, 8)])])
    )
    # 3: sfx echo (single event, not duplicated)
    out.append(
        ledger("f03", 5.0, events=[
            ev("E1", "sfx", [t(0.0, 0.3), t(0.6, 0.9)], text="glass break with echo", track_id="T1"),
        ], tracks=[tr("T1", "sfx", [t(0, 0.3), t(0.6, 0.9)])])
    )
    # 4: multi-speaker overlap
    out.append(
        ledger("f04", 6.0, events=[
            ev("E1", "speech", [t(0.0, 3.0)], text="speaker one talks", track_id="T1"),
            ev("E2", "speech", [t(1.0, 4.0)], text="speaker two overlaps", track_id="T2"),
        ], tracks=[tr("T1", "speech", [t(0, 3)], identity="S1"), tr("T2", "speech", [t(1, 4)], identity="S2")])
    )
    # 5: empty scene (silence)
    out.append(ledger("f05", 4.0))
    # 6: single music only
    out.append(
        ledger("f06", 12.0, events=[ev("E1", "music", [t(0.0, 12.0)], text="ambient drone", track_id="T1")],
               tracks=[tr("T1", "music", [t(0, 12)])])
    )
    # 7: boundary at 0.0
    out.append(
        ledger("f07", 3.0, events=[ev("E1", "sfx", [t(0.0, 0.1)], text="click", track_id="T1")],
               tracks=[tr("T1", "sfx", [t(0, 0.1)])])
    )
    # 8: boundary at duration
    out.append(
        ledger("f08", 5.0, events=[ev("E1", "music", [t(0.0, 5.0)], text="full clip", track_id="T1")],
               tracks=[tr("T1", "music", [t(0, 5)])])
    )
    # 9: 0.1 grid boundary
    out.append(
        ledger("f09", 2.0, events=[ev("E1", "sfx", [t(0.1, 0.2)], text="tick", track_id="T1")],
               tracks=[tr("T1", "sfx", [t(0.1, 0.2)])])
    )
    # 10: all four types
    out.append(
        ledger("f10", 10.0, events=[
            ev("E1", "speech", [t(0.0, 1.0)], text="hi", track_id="T1"),
            ev("E2", "lys", [t(1.5, 2.5)], text="la la", track_id="T2"),
            ev("E3", "music", [t(0.0, 10.0)], text="band", track_id="T3"),
            ev("E4", "sfx", [t(3.0, 3.2)], text="boom", track_id="T4"),
        ], tracks=[tr("T1", "speech", [t(0, 1)]), tr("T2", "vocal", [t(1.5, 2.5)]), tr("T3", "music", [t(0, 10)]), tr("T4", "sfx", [t(3, 3.2)])])
    )
    # 11: long gap between events
    out.append(
        ledger("f11", 30.0, events=[ev("E1", "sfx", [t(0.0, 0.5)], text="start", track_id="T1"), ev("E2", "sfx", [t(25.0, 25.5)], text="end", track_id="T1")],
               tracks=[tr("T1", "sfx", [t(0, 0.5), t(25, 25.5)])])
    )
    # 12: low confidence
    out.append(
        ledger("f12", 5.0, events=[ev("E1", "sfx", [t(1.0, 1.5)], text="maybe thunder", track_id="T1", confidence=0.2)],
               tracks=[tr("T1", "sfx", [t(1, 1.5)], confidence=0.2)])
    )
    # 13: vocal without lyrics (ambiguous)
    out.append(
        ledger("f13", 6.0, events=[ev("E1", "music", [t(0.0, 6.0)], text="song with unclear vocals", track_id="T1")],
               tracks=[tr("T1", "music", [t(0, 6)])])
    )
    # 14: residual/ambience track
    out.append(
        ledger("f14", 8.0, events=[ev("E1", "sfx", [t(0.0, 8.0)], text="rain ambience", track_id="T1")],
               tracks=[tr("T1", "ambience", [t(0, 8)])])
    )
    # 15: very short event (1 frame)
    out.append(
        ledger("f15", 2.0, events=[ev("E1", "sfx", [t(1.0, 1.1)], text="blip", track_id="T1")],
               tracks=[tr("T1", "sfx", [t(1.0, 1.1)])])
    )
    # 16: three overlapping speakers
    out.append(
        ledger("f16", 5.0, events=[
            ev("E1", "speech", [t(0.0, 2.0)], text="a", track_id="T1"),
            ev("E2", "speech", [t(0.5, 2.5)], text="b", track_id="T2"),
            ev("E3", "speech", [t(1.0, 3.0)], text="c", track_id="T3"),
        ], tracks=[tr("T1", "speech", [t(0, 2)], identity="S1"), tr("T2", "speech", [t(0.5, 2.5)], identity="S2"), tr("T3", "speech", [t(1, 3)], identity="S3")])
    )
    # 17: music with multiple sections
    out.append(
        ledger("f17", 12.0, events=[
            ev("E1", "music", [t(0.0, 4.0)], text="intro", track_id="T1"),
            ev("E2", "music", [t(4.0, 8.0)], text="verse", track_id="T1"),
            ev("E3", "music", [t(8.0, 12.0)], text="chorus", track_id="T1"),
        ], tracks=[tr("T1", "music", [t(0, 4), t(4, 8), t(8, 12)])])
    )
    # 18: speech with verbatim=False (paraphrase)
    out.append(
        ledger("f18", 4.0, events=[ev("E1", "speech", [t(0.0, 2.0)], text="someone greets", track_id="T1", verbatim=False)],
               tracks=[tr("T1", "speech", [t(0, 2)])])
    )
    # 19: sfx at very end
    out.append(
        ledger("f19", 10.0, events=[ev("E1", "sfx", [t(9.5, 10.0)], text="door slam", track_id="T1")],
               tracks=[tr("T1", "sfx", [t(9.5, 10)])])
    )
    # 20: dense overlap of all types
    out.append(
        ledger("f20", 8.0, events=[
            ev("E1", "music", [t(0.0, 8.0)], text="loud band", track_id="T1"),
            ev("E2", "speech", [t(1.0, 4.0)], text="announcement", track_id="T2"),
            ev("E3", "lys", [t(4.5, 6.5)], text="singing", track_id="T3"),
            ev("E4", "sfx", [t(2.0, 2.3)], text="crash", track_id="T4"),
            ev("E5", "sfx", [t(5.0, 5.2)], text="whistle", track_id="T5"),
        ], tracks=[tr("T1", "music", [t(0, 8)]), tr("T2", "speech", [t(1, 4)]), tr("T3", "vocal", [t(4.5, 6.5)]), tr("T4", "sfx", [t(2, 2.3)]), tr("T5", "sfx", [t(5, 5.2)])])
    )
    return out


FIXTURES = _build_fixture_set()


@pytest.fixture(scope="module")
def fixture_set():
    return FIXTURES


def test_fixture_count():
    assert len(FIXTURES) == 20


def test_perfect_prediction_gives_perfect_metrics():
    refs = {lg.sample_id: lg for lg in FIXTURES}
    hyps = {lg.sample_id: lg.model_copy(deep=True) for lg in FIXTURES}
    corpus = evaluate_corpus(hyps, refs)
    assert corpus.macro_event_f1 == 1.0
    assert corpus.macro_event_precision == 1.0
    assert corpus.macro_event_recall == 1.0
    assert corpus.mean_onset_mae == 0.0
    assert corpus.mean_offset_mae == 0.0
    assert corpus.total_hallucination == 0
    assert corpus.total_omission == 0
    assert corpus.mean_pointer_accuracy == 1.0
    assert corpus.strict_format_success_rate == 1.0


def test_empty_scene_handled():
    empty = next(lg for lg in FIXTURES if lg.sample_id == "f05")
    sm = evaluate_sample(empty, empty)
    assert sm.n_ref == 0
    assert sm.n_hyp == 0
    assert sm.event_f1 == 1.0  # convention: empty vs empty is perfect


def test_missing_prediction_counts_as_all_omission():
    refs = {lg.sample_id: lg for lg in FIXTURES}
    sm = evaluate_sample(refs["f10"], refs["f10"].model_copy())
    # fudge: simulate missing by direct call to metrics path
    from sceneledger.eval.metrics import SampleMetrics

    # use evaluate_corpus with only refs present
    corpus = evaluate_corpus({}, refs)
    # all refs have no prediction -> all omission
    assert corpus.total_omission == sum(len(lg.events) for lg in FIXTURES)


def test_serialize_all_fixtures_round_trip():
    for lg in FIXTURES:
        xml = serialize(lg, mode="full")
        from sceneledger.models.serializer import deserialize

        back = deserialize(xml)
        assert back.model_dump() == lg.model_dump(), f"round-trip failed for {lg.sample_id}"


def test_write_jsonl_fixtures(tmp_path: Path):
    """Materialize fixtures + a perfect-prediction JSONL for the CLI smoke test."""
    refs_path = tmp_path / "references.jsonl"
    preds_path = tmp_path / "predictions.jsonl"
    with refs_path.open("w", encoding="utf-8") as f:
        for lg in FIXTURES:
            f.write(json.dumps(lg.model_dump(mode="json"), ensure_ascii=False) + "\n")
    # predictions = identical to references (perfect run)
    with preds_path.open("w", encoding="utf-8") as f:
        for lg in FIXTURES:
            f.write(json.dumps(lg.model_dump(mode="json"), ensure_ascii=False) + "\n")
    corpus = evaluate_corpus(preds_path, refs_path)
    assert corpus.macro_event_f1 == 1.0
    assert corpus.n_samples == 20
