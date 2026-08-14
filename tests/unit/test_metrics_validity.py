"""Validity checks for raw parser evidence used by corpus metrics."""

from __future__ import annotations

import json

import pytest
from fixtures.factory import ev, ledger, t, tr

from sceneledger.cli.evaluate import main as evaluate_main
from sceneledger.data.experiment_data import file_sha256
from sceneledger.eval.metrics import evaluate_corpus, load_inference_report


def _fixture():
    return ledger(
        "sample-1",
        2.0,
        events=[ev("E1", "sfx", [t(0.2, 0.5)], text="click", track_id="T1")],
        tracks=[tr("T1", "sfx", [t(0.2, 0.5)])],
    )


def test_parser_report_controls_format_metric_even_for_perfect_ledger(tmp_path) -> None:
    reference = _fixture()
    report = tmp_path / "infer_report.json"
    report.write_text(
        json.dumps(
            {
                "n_samples": 1,
                "strict_format_success_rate": 0.0,
                "samples": [
                    {
                        "sample_id": reference.sample_id,
                        "strict_format_success": False,
                        "warnings": ["tolerant parser repaired malformed output"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_corpus(
        {reference.sample_id: reference.model_copy(deep=True)},
        {reference.sample_id: reference},
        inference_report=report,
    )

    assert result.macro_event_f1 == 1.0
    assert result.strict_format_success_rate == 0.0
    assert result.format_status_complete is True
    assert result.samples[0]["warnings"] == [
        "tolerant parser repaired malformed output"
    ]


def test_format_metric_is_unavailable_without_raw_parser_evidence() -> None:
    reference = _fixture()
    result = evaluate_corpus(
        {reference.sample_id: reference.model_copy(deep=True)},
        {reference.sample_id: reference},
    )

    assert result.strict_format_success_rate is None
    assert result.format_status_complete is False
    assert result.n_format_status_known == 0
    assert result.n_format_status_missing == 1


def test_inference_report_rejects_duplicate_ids(tmp_path) -> None:
    report = tmp_path / "duplicate.json"
    row = {
        "sample_id": "sample-1",
        "strict_format_success": True,
        "warnings": [],
    }
    report.write_text(json.dumps({"samples": [row, row]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_inference_report(report)


def test_inference_report_must_cover_the_reference_set() -> None:
    reference = _fixture()
    report = {
        "samples": [
            {
                "sample_id": "another-sample",
                "strict_format_success": True,
                "warnings": [],
            }
        ]
    }

    with pytest.raises(ValueError, match="do not equal reference IDs"):
        evaluate_corpus(
            {reference.sample_id: reference.model_copy(deep=True)},
            {reference.sample_id: reference},
            inference_report=report,
        )


def test_inference_report_rejects_inconsistent_summary() -> None:
    report = {
        "n_samples": 1,
        "strict_format_success_rate": 1.0,
        "samples": [
            {
                "sample_id": "sample-1",
                "strict_format_success": False,
                "warnings": [],
            }
        ],
    }

    with pytest.raises(ValueError, match="inconsistent"):
        load_inference_report(report)


def test_cli_rejects_prediction_modified_after_inference(tmp_path) -> None:
    reference = _fixture()
    prediction = tmp_path / "prediction.jsonl"
    references = tmp_path / "references.jsonl"
    report = tmp_path / "infer_report.json"
    for path in (prediction, references):
        path.write_text(
            json.dumps(reference.model_dump(mode="json")) + "\n",
            encoding="utf-8",
        )
    report.write_text(
        json.dumps(
            {
                "schema_version": "sceneledger-inference-report-v1",
                "n_samples": 1,
                "strict_format_success_rate": 1.0,
                "prediction_sha256": file_sha256(prediction),
                "samples": [
                    {
                        "sample_id": reference.sample_id,
                        "strict_format_success": True,
                        "warnings": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    prediction.write_text(prediction.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prediction file hash does not match"):
        evaluate_main(
            [
                "--prediction",
                str(prediction),
                "--reference",
                str(references),
                "--inference-report",
                str(report),
            ]
        )
