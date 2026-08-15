from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import sceneledger.eval.model_review as review
from sceneledger.data.experiment_data import file_sha256
from sceneledger.data.manifests import ManifestEntry, write_manifest
from sceneledger.eval.metrics import INFERENCE_REPORT_SCHEMA_VERSION


def _ledger(sample_id: str, text: str) -> dict:
    return {
        "schema_version": "0.2.0",
        "sample_id": sample_id,
        "duration_sec": 2.0,
        "time_resolution_sec": 0.1,
        "tracks": [
            {
                "id": "T1",
                "kind": "sfx",
                "spans": [{"start_sec": 0.2, "end_sec": 1.5}],
                "confidence": 1.0,
            }
        ],
        "events": [
            {
                "id": "E001",
                "type": "sfx",
                "track_id": "T1",
                "spans": [{"start_sec": 0.2, "end_sec": 1.5}],
                "text": text,
                "confidence": 1.0,
            }
        ],
    }


def _entry(sample_id: str, template: str) -> ManifestEntry:
    return ManifestEntry(
        scene={
            "scene_id": sample_id,
            "seed": 1,
            "duration": 2.0,
            "template": template,
            "sources": [],
        },
        mixture_path=f"audio/{sample_id}.wav",
        stem_paths={},
        mixture_hash="mix",
        dry_mixture_hash="dry",
        stem_hashes={},
        activity_hashes={},
        target_ledger=_ledger(sample_id, "reference"),
        sample_rate=16000,
    )


def _write_ledgers(path: Path, payloads: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_blinded_review_round_trip_and_tamper_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    sample_ids = [f"test-{index}" for index in range(4)]
    entries = [
        _entry(sample_id, "speech_sfx" if index % 2 else "multi_sfx")
        for index, sample_id in enumerate(sample_ids)
    ]
    manifest = tmp_path / "test.jsonl"
    write_manifest(manifest, entries)
    for sample_id in sample_ids:
        audio = tmp_path / "audio" / f"{sample_id}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"test-audio")

    zero_predictions = tmp_path / "zero.jsonl"
    tuned_predictions = tmp_path / "tuned.jsonl"
    _write_ledgers(
        zero_predictions, [_ledger(sample_id, "generic sound") for sample_id in sample_ids]
    )
    _write_ledgers(
        tuned_predictions,
        [_ledger(sample_id, "specific supported sound") for sample_id in sample_ids],
    )
    zero_report = tmp_path / "zero_report.json"
    tuned_report = tmp_path / "tuned_report.json"
    for path, predictions in (
        (zero_report, zero_predictions),
        (tuned_report, tuned_predictions),
    ):
        _write_json(
            path,
            {
                "schema_version": INFERENCE_REPORT_SCHEMA_VERSION,
                "dataset_id": "dataset-1",
                "expected_split": "test",
                "prediction_sha256": file_sha256(predictions),
                "n_samples": 4,
                    "samples": [
                        {
                            "sample_id": sample_id,
                            "strict_format_success": True,
                            "explicit_track_ids_complete": True,
                        }
                        for sample_id in sample_ids
                    ],
            },
        )
    validity = tmp_path / "validity.json"
    _write_json(
        validity,
        {
            "schema_version": "sceneledger-evaluation-validity-v1",
            "pass": True,
            "status": "certified_generalization",
            "dataset_id": "dataset-1",
            "artifacts": {
                "inference_report": {"sha256": file_sha256(tuned_report)}
            },
        },
    )
    split_contract = tmp_path / "split.json"
    data_gate = tmp_path / "gate.json"
    _write_json(split_contract, {})
    _write_json(data_gate, {})
    monkeypatch.setattr(
        review,
        "require_experiment_data_summary",
        lambda *_: {"dataset_id": "dataset-1"},
    )
    monkeypatch.setattr(
        review,
        "require_split_manifest",
        lambda *_: {"dataset_id": "dataset-1"},
    )
    monkeypatch.setattr(review, "require_ledger_split", lambda *_args, **_kwargs: {})

    rows, metadata, key = review.prepare_model_review(
        manifest_path=manifest,
        audio_base=tmp_path,
        zero_predictions_path=zero_predictions,
        zero_inference_report_path=zero_report,
        tuned_predictions_path=tuned_predictions,
        tuned_inference_report_path=tuned_report,
        validity_audit_path=validity,
        split_contract_path=split_contract,
        data_gate_summary_path=data_gate,
        sample_count=4,
        seed="unit-test",
    )
    assignments = {item["task_id"]: item for item in key["assignments"]}
    for row in rows:
        row["reviewer"] = "reviewer-1"
        row["reviewed_at_utc"] = "2026-08-15T00:00:00Z"
        tuned_side = "a" if assignments[row["task_id"]]["arm_a"] == "b3_tuned" else "b"
        zero_side = "b" if tuned_side == "a" else "a"
        for side, score, errors in ((tuned_side, "5", "0"), (zero_side, "3", "1")):
            row[f"{side}_semantic_support_1_5"] = score
            row[f"{side}_completeness_1_5"] = score
            row[f"{side}_temporal_alignment_1_5"] = score
            row[f"{side}_source_attribution_1_5_or_na"] = score
            row[f"{side}_hallucination_count"] = errors
            row[f"{side}_omission_count"] = errors
        row["preference_a_b_tie"] = tuned_side
        row["notes"] = ""

    csv_path = tmp_path / "review.csv"
    metadata_path = tmp_path / "review.metadata.json"
    key_path = tmp_path / "review.key.json"
    review.write_model_review(
        rows,
        metadata,
        key,
        csv_path=csv_path,
        metadata_path=metadata_path,
        key_path=key_path,
    )
    reviewer2_rows = [dict(row, reviewer="reviewer-2") for row in rows]
    reviewer2_path = tmp_path / "reviewer2.csv"
    with reviewer2_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(reviewer2_rows)
    summary = review.summarize_model_review(
        review_csv_path=[csv_path, reviewer2_path],
        metadata_path=metadata_path,
        key_path=key_path,
    )
    assert summary["pass"] is True
    assert summary["go_for_scale"] is True
    assert summary["n_completed_reviews"] == 2
    assert summary["n_reviewers"] == 2
    assert summary["preference_sample_consensus"]["b3_tuned"] == 4
    assert summary["preference_sample_consensus"]["pairwise_reviewer_agreement"] == 1.0
    assert summary["arms"]["b3_tuned"]["mean_semantic_support"] == 5.0
    assert summary["paired_delta_tuned_minus_zero"]["mean_semantic_support"] == 2.0

    rows[0]["candidate_a_json"] = "tampered"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="immutable model review fields changed"):
        review.summarize_model_review(
            review_csv_path=csv_path,
            metadata_path=metadata_path,
            key_path=key_path,
        )
