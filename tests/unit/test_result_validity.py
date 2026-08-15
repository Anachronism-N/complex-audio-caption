from __future__ import annotations

import json
from pathlib import Path

import yaml

import sceneledger.eval.result_validity as validity
from sceneledger.data.manifests import ManifestEntry, write_manifest
from sceneledger.eval.metrics import METRICS_SCHEMA_VERSION


def _entry(sample_id: str, source_group: str) -> ManifestEntry:
    return ManifestEntry(
        scene={
            "scene_id": sample_id,
            "seed": 1,
            "duration": 1.0,
            "template": "test",
            "sources": [
                {
                    "source_id": "S1",
                    "kind": "sfx",
                    "path": f"audio/{source_group}.wav",
                    "source_group": source_group,
                }
            ],
        },
        mixture_path=f"audio/{sample_id}.wav",
        stem_paths={},
        mixture_hash="mix",
        dry_mixture_hash="dry",
        stem_hashes={},
        activity_hashes={},
        target_ledger={},
        sample_rate=16000,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_certifies_disjoint_complete_contracted_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    train_manifest = tmp_path / "train.jsonl"
    test_manifest = tmp_path / "test.jsonl"
    write_manifest(train_manifest, [_entry("train-1", "speaker-train")])
    write_manifest(test_manifest, [_entry("test-1", "recording-test")])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"manifest_path": str(train_manifest), "pre_split": True},
                "train": {"seed": 7},
                "experiment_contract": {"dataset_id": "dataset-1"},
            }
        ),
        encoding="utf-8",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"].update(
        {
            "expected_split": "train",
            "require_human_audit": True,
            "human_audit_summary_path": str(tmp_path / "human.json"),
        }
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_json(tmp_path / "human.json", {})
    metrics_path = tmp_path / "metrics.json"
    inference_path = tmp_path / "inference.json"
    _write_json(
        metrics_path,
        {
            "schema_version": METRICS_SCHEMA_VERSION,
            "n_samples": 1,
            "strict_format_success_rate": 1.0,
            "format_status_complete": True,
            "n_format_status_known": 1,
            "n_format_status_missing": 0,
            "macro_event_precision": 0.5,
            "macro_event_recall": 0.5,
            "macro_event_f1": 0.5,
            "macro_caption_token_f1": 0.4,
            "macro_seg_f1_100ms": 0.5,
            "mean_onset_mae": 0.1,
            "mean_offset_mae": 0.2,
            "mean_onset_p90": 0.1,
            "mean_offset_p90": 0.2,
            "macro_tolerance_acc_010": 0.5,
            "macro_tolerance_acc_025": 0.5,
            "macro_tolerance_acc_050": 0.5,
            "macro_tolerance_acc_100": 1.0,
            "total_hallucination": 0,
            "total_omission": 0,
            "mean_source_count_mae": 0.0,
            "mean_pointer_accuracy": 1.0,
            "experiment_contract": {"dataset_id": "dataset-1", "split": "test"},
            "inference_evidence": {
                "sha256": "patched-below",
                "prediction_sha256": "bound",
            },
            "samples": [
                {
                    "sample_id": "test-1",
                    "event_precision": 0.5,
                    "event_recall": 0.5,
                    "event_f1": 0.5,
                    "caption_token_f1": 0.4,
                    "seg_f1_100ms": 0.5,
                    "onset_mae": 0.1,
                    "offset_mae": 0.2,
                    "onset_p90": 0.1,
                    "offset_p90": 0.2,
                    "tolerance_acc_010": 0.5,
                    "tolerance_acc_025": 0.5,
                    "tolerance_acc_050": 0.5,
                    "tolerance_acc_100": 1.0,
                    "hallucination": 0,
                    "omission": 0,
                    "source_count_mae": 0.0,
                    "pointer_accuracy": 1.0,
                    "strict_format_success": True,
                }
            ],
        },
    )
    _write_json(
        inference_path,
        {
            "schema_version": "sceneledger-inference-report-v1",
            "dataset_id": "dataset-1",
            "expected_split": "test",
            "prediction_sha256": "bound",
            "n_samples": 1,
            "strict_format_success_rate": 1.0,
            "samples": [
                {"sample_id": "test-1", "strict_format_success": True}
            ],
        },
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["inference_evidence"]["sha256"] = validity.file_sha256(inference_path)
    _write_json(metrics_path, metrics)
    monkeypatch.setattr(
        validity,
        "require_experiment_data_summary",
        lambda *_: {"dataset_id": "dataset-1"},
    )
    monkeypatch.setattr(validity, "require_split_manifest", lambda *_: {})
    monkeypatch.setattr(validity, "require_human_audit_summary", lambda *_args, **_kwargs: {})

    report = validity.audit_evaluation_result(
        train_config_path=config_path,
        eval_manifest_path=test_manifest,
        metrics_path=metrics_path,
        inference_report_path=inference_path,
        repo_root=tmp_path,
        split_contract_path=tmp_path / "split.json",
        data_gate_summary_path=tmp_path / "gate.json",
    )

    assert report["pass"] is True
    assert report["status"] == "certified_generalization"
    assert report["claim_scope"] == "paper_eligible"
    assert report["dataset_id"] == "dataset-1"
    assert report["counts"]["reported_seen_during_training"] == 0


def test_rejects_current_v6_index_slice_claim() -> None:
    repo = Path(__file__).resolve().parents[2]
    report = validity.audit_evaluation_result(
        train_config_path="configs/model/b3_real_v6_3k.yaml",
        eval_manifest_path="data/derived/real_mix_v6/manifest_compat.jsonl",
        metrics_path="reports/b3_real_v6_3k_heldout_metrics.json",
        inference_report_path="reports/b3_real_v6_3k_heldout_infer_report.json",
        repo_root=repo,
    )

    assert report["pass"] is False
    assert report["counts"]["reported_seen_during_training"] == 15
    assert report["metric_subgroups"]["seen_during_training"]["mean_event_f1"] == 1.0
    assert report["metric_subgroups"]["unseen_by_sample_id"]["mean_event_f1"] == 0.866667
    assert "raw_source_identity_auditable" in report["failed_checks"]


def test_rejects_current_v6k_breakthrough_claim() -> None:
    repo = Path(__file__).resolve().parents[2]
    report = validity.audit_evaluation_result(
        train_config_path="configs/model/b3_real_v6k_3k.yaml",
        eval_manifest_path="data/derived/real_mix_v6_1k/manifest_compat.jsonl",
        metrics_path="reports/b3_real_v6k_3k_heldout_metrics.json",
        inference_report_path="reports/b3_real_v6k_3k_heldout_infer_report.json",
        repo_root=repo,
    )

    assert report["pass"] is False
    assert report["claim_scope"] == "diagnostic_only"
    assert report["counts"]["reported_seen_during_training"] == 64
    assert report["counts"]["reported_unseen_by_sample_id"] == 36
    assert report["metric_subgroups"]["seen_during_training"]["mean_event_f1"] == 0.989583
    assert report["metric_subgroups"]["unseen_by_sample_id"]["mean_event_f1"] == 0.935185
    assert "metrics_parser_evidence_identical" in report["failed_checks"]
    assert "current_semantic_metric_schema_valid" in report["failed_checks"]
    assert report["headline_metrics"]["caption_token_f1"] is None
