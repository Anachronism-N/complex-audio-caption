"""Validity checks for gated metrics and raw parse-report propagation."""

from __future__ import annotations

import json

from fixtures.factory import ev, ledger, t

from sceneledger.eval.metrics import evaluate_corpus


def test_text_gate_changes_type_time_only_true_positive():
    ref = ledger(
        "sample",
        2.0,
        events=[ev("E1", "sfx", [t(0.0, 1.0)], "glass breaks")],
    )
    hyp = ledger(
        "sample",
        2.0,
        events=[ev("E2", "sfx", [t(0.0, 1.0)], "a dog barks")],
    )
    temporal = evaluate_corpus({"sample": hyp}, {"sample": ref})
    gated = evaluate_corpus(
        {"sample": hyp}, {"sample": ref}, min_text_similarity=0.1
    )
    assert temporal.macro_event_f1 == 1.0
    assert temporal.macro_zero_text_match_rate == 1.0
    assert gated.macro_event_f1 == 0.0
    assert gated.min_text_similarity == 0.1


def test_inference_parse_report_controls_format_success(tmp_path):
    ref = ledger(
        "sample",
        2.0,
        events=[ev("E1", "sfx", [t(0.0, 1.0)], "impact")],
    )
    report = tmp_path / "infer_report.json"
    report.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample",
                        "strict_format_success": False,
                        "warnings": ["timestamp clipped"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_corpus(
        {"sample": ref}, {"sample": ref}, parse_reports=report
    )
    assert result.strict_format_success_rate == 0.0
    assert result.samples[0]["warnings"] == ["timestamp clipped"]
