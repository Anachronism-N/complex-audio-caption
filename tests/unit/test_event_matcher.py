"""Unit tests for Hungarian event matching and permutation invariance."""

from __future__ import annotations

import random

from fixtures.factory import ev, t

from sceneledger.eval.event_matcher import match_events, token_f1


def _events():
    return [
        ev("E1", "speech", [t(0.0, 1.0)], text="hello world", track_id="T1"),
        ev("E2", "music", [t(2.0, 4.0)], text="rock song", track_id="T2"),
        ev("E3", "sfx", [t(5.0, 5.5)], text="glass break", track_id="T3"),
    ]


def test_perfect_match():
    refs = _events()
    hyps = [e.model_copy() for e in refs]
    matches = match_events(refs, hyps)
    matched = [m for m in matches if m.is_match]
    assert len(matched) == 3
    assert all(m.tiou == 1.0 for m in matched)


def test_permutation_invariant():
    refs = _events()
    rng = random.Random(42)
    hyps = list(refs)
    rng.shuffle(hyps)
    matches = match_events(refs, hyps)
    matched_pairs = {(m.ref_id, m.hyp_id) for m in matches if m.is_match}
    assert matched_pairs == {("E1", "E1"), ("E2", "E2"), ("E3", "E3")}


def test_omission_and_hallucination():
    refs = _events()
    # drop E2 (omission), add a spurious sfx (hallucination)
    hyps = [
        refs[0].model_copy(),
        refs[2].model_copy(),
        ev("E9", "sfx", [t(8.0, 8.5)], text="dog bark", track_id="T9"),
    ]
    matches = match_events(refs, hyps)
    omissions = [m for m in matches if m.hyp_id is None]
    halluc = [m for m in matches if m.ref_id is None]
    assert len(omissions) == 1
    assert omissions[0].ref_id == "E2"
    assert len(halluc) == 1
    assert halluc[0].hyp_id == "E9"


def test_type_gate_prevents_cross_type_match():
    # same time, different type -> must NOT match
    refs = [ev("E1", "speech", [t(0.0, 1.0)], text="hi")]
    hyps = [ev("E2", "music", [t(0.0, 1.0)], text="hi")]
    matches = match_events(refs, hyps)
    assert not any(m.is_match for m in matches)


def test_tiou_threshold_gate():
    refs = [ev("E1", "sfx", [t(0.0, 1.0)], text="x")]
    # tIoU = 0.1/1.1 ~ 0.09 < 0.3 threshold
    hyps = [ev("E2", "sfx", [t(0.9, 1.9)], text="x")]
    matches = match_events(refs, hyps, tiou_threshold=0.3)
    assert not any(m.is_match for m in matches)


def test_token_f1_hand_calc():
    assert token_f1("hello world", "world hello") == 1.0
    # ta={a,b,c} (3), tb={a,b} (2), inter=2: prec=1.0, rec=2/3, f1=0.8
    assert token_f1("a b c", "a b") == 0.8


def test_track_pointer_agreement_reported():
    refs = [ev("E1", "speech", [t(0.0, 1.0)], text="hi", track_id="T1")]
    hyps = [ev("E2", "speech", [t(0.0, 1.0)], text="hi", track_id="T2")]
    matches = match_events(refs, hyps)
    m = next(x for x in matches if x.is_match)
    assert m.track_match is False


def test_text_gate_rejects_semantically_unsupported_match():
    refs = [ev("E1", "sfx", [t(0.0, 1.0)], text="glass breaks")]
    hyps = [ev("E2", "sfx", [t(0.0, 1.0)], text="a dog barks")]
    legacy = match_events(refs, hyps)
    gated = match_events(refs, hyps, min_text_similarity=0.1)
    assert any(match.is_match for match in legacy)
    assert not any(match.is_match for match in gated)
