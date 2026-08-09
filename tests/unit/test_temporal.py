"""Unit tests for temporal metrics with hand-computed toy values."""

from __future__ import annotations

from fixtures.factory import ev, t

from sceneledger.eval.temporal import (
    boundary_mae,
    multi_span_iou,
    seg_f1,
    tolerance_accuracy,
)


def test_multi_span_iou_identical():
    spans = [t(0.0, 1.0), t(2.0, 3.0)]
    assert multi_span_iou(spans, spans) == 1.0


def test_multi_span_iou_disjoint_is_zero():
    assert multi_span_iou([t(0.0, 1.0)], [t(2.0, 3.0)]) == 0.0


def test_multi_span_iou_hand_calc():
    # union of A = [0,2], B=[1,3]; intersect [1,2]=1, union=3 -> 1/3
    assert multi_span_iou([t(0.0, 2.0)], [t(1.0, 3.0)]) == round(1 / 3, 6)


def test_multi_span_iou_with_overlap_in_one_of_two_spans():
    # A = [0,1] U [2,3] (len 2); B = [0.5,1.5] (len 1); inter [0.5,1]=0.5; union=2.5
    assert multi_span_iou([t(0.0, 1.0), t(2.0, 3.0)], [t(0.5, 1.5)]) == round(0.5 / 2.5, 6)


def test_boundary_mae_hand_calc():
    ref = ev("E1", "sfx", [t(1.0, 2.0)], text="x")
    hyp = ev("E2", "sfx", [t(1.2, 2.5)], text="x")
    berr = boundary_mae([(ref, hyp)])
    assert berr.onset_mae == 0.2
    assert berr.offset_mae == 0.5


def test_tolerance_accuracy_hand_calc():
    ref = ev("E1", "sfx", [t(1.0, 2.0)], text="x")
    h1 = ev("E2", "sfx", [t(1.1, 2.1)], text="x")  # within 0.2
    h2 = ev("E3", "sfx", [t(1.5, 2.5)], text="x")  # onset off by 0.5
    acc_025 = tolerance_accuracy([(ref, h1), (ref, h2)], collar_seconds=0.25)
    # h1 within 0.25, h2 onset off by 0.5 > 0.25 -> 1/2
    assert acc_025 == 0.5


def test_seg_f1_perfect():
    e = ev("E1", "sfx", [t(0.0, 1.0)], text="x")
    p, r, f1 = seg_f1([e], [e])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_seg_f1_half_overlap_hand_calc():
    # ref [0,1], hyp [0.5,1.5] at 0.1 grid -> 5 frames overlap, 5 fp, 5 fn
    ref = ev("E1", "sfx", [t(0.0, 1.0)], text="x")
    hyp = ev("E2", "sfx", [t(0.5, 1.5)], text="x")
    p, r, f1 = seg_f1([ref], [hyp])
    # tp=5, fp=5, fn=5 -> p=r=0.5, f1=0.5
    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_seg_f1_different_type_no_overlap():
    ref = ev("E1", "sfx", [t(0.0, 1.0)], text="x")
    hyp = ev("E2", "music", [t(0.0, 1.0)], text="x")
    p, r, f1 = seg_f1([ref], [hyp])
    # sfx: tp=0,fp=0,fn=10 ; music: tp=0,fp=10,fn=0 -> p=0/(0+10)=0, r=0/(0+10)=0
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0
