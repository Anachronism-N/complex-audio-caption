"""CPU-only replay of committed raw generations with the current evaluator.

This recovers missing modern diagnostics from historical inference reports. It
does not retroactively freeze a test split or turn a contaminated experiment
into a paper result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from sceneledger.data.experiment_data import file_sha256
from sceneledger.data.manifests import ManifestEntry, read_manifest
from sceneledger.data.schema import Ledger
from sceneledger.eval.metrics import (
    METRICS_SCHEMA_VERSION,
    CorpusMetrics,
    evaluate_corpus,
    load_inference_report,
    validate_metrics_artifact,
)
from sceneledger.eval.parser import parse_caption_output
from sceneledger.eval.result_validity import reconstruct_training_entries

FORENSIC_REPLAY_SCHEMA_VERSION = "sceneledger-forensic-replay-v1"

_HEADLINE_FIELDS = (
    "n_samples",
    "strict_format_success_rate",
    "macro_event_precision",
    "macro_event_recall",
    "macro_event_f1",
    "macro_caption_token_f1",
    "macro_seg_f1_100ms",
    "mean_onset_mae",
    "mean_offset_mae",
    "macro_tolerance_acc_010",
    "total_hallucination",
    "total_omission",
    "mean_source_count_mae",
    "mean_pointer_accuracy",
)


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _headline(metrics: CorpusMetrics | dict) -> dict[str, Any]:
    payload = metrics.to_dict() if isinstance(metrics, CorpusMetrics) else metrics
    return {field: payload.get(field) for field in _HEADLINE_FIELDS}


def _subset_report(
    sample_ids: list[str], statuses: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = [statuses[sample_id] for sample_id in sample_ids]
    return {
        "n_samples": len(rows),
        "strict_format_success_rate": round(
            sum(bool(row["strict_format_success"]) for row in rows)
            / max(1, len(rows)),
            4,
        ),
        "samples": rows,
    }


def _unauditable_source_count(entries: list[ManifestEntry]) -> int:
    count = 0
    for entry in entries:
        for source in entry.scene.get("sources", []):
            path = str(source.get("path", "")).replace("\\", "/")
            group = source.get("source_group")
            if (not path or path.startswith("real:")) and group in (None, ""):
                count += 1
    return count


def replay_raw_inference(
    *,
    train_config_path: str | Path,
    manifest_path: str | Path,
    inference_report_path: str | Path,
    repo_root: str | Path = ".",
    original_metrics_path: str | Path | None = None,
) -> tuple[dict[str, Ledger], dict[str, Ledger], dict, dict]:
    """Reparse raw text and return predictions, references, metrics and report."""
    root = Path(repo_root).expanduser().resolve()
    config_path = _resolve(train_config_path, root)
    eval_manifest_path = _resolve(manifest_path, root)
    inference_path = _resolve(inference_report_path, root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    train_manifest_path = _resolve(config["data"]["manifest_path"], root)
    train_entries = read_manifest(train_manifest_path)
    actual_training = reconstruct_training_entries(config, train_entries)
    actual_training_ids = {
        str(entry.scene.get("scene_id", "")) for entry in actual_training
    }

    entries = read_manifest(eval_manifest_path)
    entries_by_id: dict[str, ManifestEntry] = {}
    for entry in entries:
        sample_id = str(entry.scene.get("scene_id", ""))
        if not sample_id or sample_id in entries_by_id:
            raise ValueError(f"evaluation manifest has invalid sample ID: {sample_id!r}")
        entries_by_id[sample_id] = entry

    original_inference, original_statuses = load_inference_report(inference_path)
    original_rows = {
        str(row["sample_id"]): row for row in original_inference["samples"]
    }
    ordered_ids = [str(row["sample_id"]) for row in original_inference["samples"]]
    unknown = sorted(set(ordered_ids) - set(entries_by_id))
    if unknown:
        raise ValueError(f"inference report IDs are absent from manifest: {unknown[:20]}")

    predictions: dict[str, Ledger] = {}
    references: dict[str, Ledger] = {}
    current_statuses: dict[str, dict[str, Any]] = {}
    parser_disagreements: list[dict[str, Any]] = []
    for sample_id in ordered_ids:
        entry = entries_by_id[sample_id]
        raw_text = original_rows[sample_id].get("raw_text")
        if not isinstance(raw_text, str) or not raw_text:
            raise ValueError(f"raw_text is missing for sample {sample_id}")
        prediction, parse_report = parse_caption_output(
            raw_text,
            sample_id,
            float(entry.scene["duration"]),
        )
        predictions[sample_id] = prediction
        references[sample_id] = Ledger.model_validate(entry.target_ledger)
        current_statuses[sample_id] = {
            "sample_id": sample_id,
            "strict_format_success": parse_report.strict_format_success,
            "warnings": list(parse_report.warnings),
        }
        original_status = original_statuses[sample_id]["strict_format_success"]
        if original_status != parse_report.strict_format_success:
            parser_disagreements.append(
                {
                    "sample_id": sample_id,
                    "original": original_status,
                    "current": parse_report.strict_format_success,
                }
            )

    current_inference = _subset_report(ordered_ids, current_statuses)
    all_metrics = evaluate_corpus(
        predictions, references, inference_report=current_inference
    )
    metrics_payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        **all_metrics.to_dict(),
        "forensic_only": True,
        "forensic_source": {
            "inference_report": _display(inference_path, root),
            "inference_report_sha256": file_sha256(inference_path),
            "manifest": _display(eval_manifest_path, root),
            "manifest_sha256": file_sha256(eval_manifest_path),
        },
    }
    validate_metrics_artifact(metrics_payload)

    seen_ids = [sample_id for sample_id in ordered_ids if sample_id in actual_training_ids]
    unseen_ids = [
        sample_id for sample_id in ordered_ids if sample_id not in actual_training_ids
    ]

    def _evaluate_subset(sample_ids: list[str]) -> CorpusMetrics:
        return evaluate_corpus(
            {sample_id: predictions[sample_id] for sample_id in sample_ids},
            {sample_id: references[sample_id] for sample_id in sample_ids},
            inference_report=_subset_report(sample_ids, current_statuses),
        )

    original_metrics: dict[str, Any] | None = None
    original_metrics_artifact: dict[str, str] | None = None
    if original_metrics_path is not None:
        original_path = _resolve(original_metrics_path, root)
        original_metrics = json.loads(original_path.read_text(encoding="utf-8"))
        original_metrics_artifact = {
            "path": _display(original_path, root),
            "sha256": file_sha256(original_path),
        }

    reported_entries = [entries_by_id[sample_id] for sample_id in ordered_ids]
    unauditable_sources = _unauditable_source_count(reported_entries)
    blockers = [
        "posthoc_report_subset_was_not_a_frozen_test_manifest",
        "forensic_replay_does_not_establish_source_disjointness",
    ]
    if unauditable_sources:
        blockers.append("missing_source_identity_cannot_be_recovered_from_raw_text")
    if seen_ids:
        blockers.append("reported_samples_were_seen_during_training")
    report = {
        "schema_version": FORENSIC_REPLAY_SCHEMA_VERSION,
        "paper_eligible": False,
        "claim_scope": "diagnostic_only",
        "publication_blockers": blockers,
        "artifacts": {
            "train_config": {
                "path": _display(config_path, root),
                "sha256": file_sha256(config_path),
            },
            "train_manifest": {
                "path": _display(train_manifest_path, root),
                "sha256": file_sha256(train_manifest_path),
            },
            "evaluation_source_manifest": {
                "path": _display(eval_manifest_path, root),
                "sha256": file_sha256(eval_manifest_path),
            },
            "inference_report": {
                "path": _display(inference_path, root),
                "sha256": file_sha256(inference_path),
            },
            "original_metrics": original_metrics_artifact,
        },
        "counts": {
            "reported": len(ordered_ids),
            "seen_during_training": len(seen_ids),
            "unseen_by_sample_id": len(unseen_ids),
            "unauditable_reported_sources": unauditable_sources,
        },
        "parser_replay": {
            "n_status_disagreements": len(parser_disagreements),
            "disagreements": parser_disagreements,
            "original_strict_format_success_rate": original_inference.get(
                "strict_format_success_rate"
            ),
            "current_strict_format_success_rate": current_inference[
                "strict_format_success_rate"
            ],
        },
        "original_headline_metrics": (
            _headline(original_metrics) if original_metrics is not None else None
        ),
        "replayed_headline_metrics": _headline(all_metrics),
        "subgroups": {
            "seen_during_training": _headline(_evaluate_subset(seen_ids)),
            "unseen_by_sample_id": _headline(_evaluate_subset(unseen_ids)),
        },
        "sample_ids": {
            "seen_during_training": seen_ids,
            "unseen_by_sample_id": unseen_ids,
        },
        "interpretation": (
            "Replayed metrics recover current parser and caption diagnostics only. "
            "They cannot retroactively certify a post-hoc or source-opaque split."
        ),
    }
    return predictions, references, metrics_payload, report


__all__ = ["replay_raw_inference"]
