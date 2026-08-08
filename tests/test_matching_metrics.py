from __future__ import annotations

import pytest

from sceneledger.matching import temporal_iou, token_f1
from sceneledger.metrics import evaluate_corpus
from sceneledger.types import Event, Ledger, Span, Track


def ledger(sample_id: str, start: float, end: float, text: str, track: str = "T1") -> Ledger:
    return Ledger(
        sample_id=sample_id,
        duration_sec=3.0,
        tracks=[Track(track, "sfx", [Span(start, end)], 1.0)],
        events=[Event("E1", "sfx", track, [Span(start, end)], text, 1.0)],
    )


def test_temporal_iou_handles_multi_span() -> None:
    left = [Span(0.0, 1.0), Span(2.0, 3.0)]
    right = [Span(0.5, 1.5), Span(2.0, 2.5)]
    assert temporal_iou(left, right) == pytest.approx(1.0 / 2.5)


def test_token_f1_supports_chinese_and_english() -> None:
    assert token_f1("狗 barking", "狗 barking") == 1.0
    assert 0 < token_f1("一只狗 barking", "狗 barking") < 1


def test_corpus_metrics() -> None:
    reference = ledger("x", 0.5, 1.5, "glass breaking")
    prediction = ledger("x", 0.6, 1.6, "glass breaking")
    metrics = evaluate_corpus([reference], [prediction])
    assert metrics.event_f1 == 1.0
    assert metrics.mean_semantic_f1 == 1.0
    assert metrics.onset_mae_sec == pytest.approx(0.1)
    assert metrics.mean_temporal_iou == pytest.approx(0.9 / 1.1)
