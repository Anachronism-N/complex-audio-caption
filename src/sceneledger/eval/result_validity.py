"""Audit whether an evaluation supports a held-out generalization claim.

Metric files alone cannot establish that an example was held out.  This module
reconstructs the exact membership used by :mod:`sceneledger.cli.train`, checks
the evaluation IDs and raw-source identities, and requires the frozen data
contract used by paper-valid experiments.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from sceneledger.data.experiment_data import (
    file_sha256,
    require_experiment_data_summary,
    require_split_manifest,
)
from sceneledger.data.human_audit import require_human_audit_summary
from sceneledger.data.manifests import ManifestEntry, read_manifest
from sceneledger.eval.metrics import validate_metrics_artifact


def _sample_id(entry: ManifestEntry) -> str:
    return str(entry.scene["scene_id"])


def _legacy_group_key(entry: ManifestEntry, key: str) -> str:
    """Mirror the historical trainer's grouping algorithm exactly."""
    if key == "source_id":
        sources = entry.scene.get("sources", [])
        if not sources:
            return "empty"
        paths = sorted(str(source.get("path", "")) for source in sources)
        return hashlib.sha1("|".join(paths).encode()).hexdigest()[:12]
    if key == "template":
        return str(entry.scene.get("template", "unknown"))
    return key


def reconstruct_training_entries(
    config: dict[str, Any], entries: list[ManifestEntry]
) -> list[ManifestEntry]:
    """Return the samples actually visited by the current training CLI."""
    data = config.get("data", {})
    if data.get("pre_split", False):
        return entries

    groups: dict[str, list[ManifestEntry]] = {}
    group_key = str(data.get("group_key", "source_id"))
    for entry in entries:
        groups.setdefault(_legacy_group_key(entry, group_key), []).append(entry)
    group_ids = sorted(groups)
    rng = random.Random(int(config.get("train", {}).get("seed", 20260808)))
    rng.shuffle(group_ids)
    val_fraction = float(data.get("val_fraction", 0.1))
    n_val = max(1, int(round(len(group_ids) * val_fraction)))
    return [entry for group in group_ids[n_val:] for entry in groups[group]]


def _source_identities(entry: ManifestEntry) -> tuple[set[str], list[str]]:
    identities: set[str] = set()
    unauditable: list[str] = []
    for source in entry.scene.get("sources", []):
        source_id = str(source.get("source_id", "unknown"))
        group = source.get("source_group")
        leakage_groups = [
            str(value)
            for value in source.get("leakage_groups", [])
            if value not in (None, "")
        ]
        if group not in (None, ""):
            identities.add(f"group:{group}")
        identities.update(f"group:{value}" for value in leakage_groups)
        if group not in (None, "") or leakage_groups:
            continue

        path = str(source.get("path", "")).replace("\\", "/")
        if path:
            identities.add(f"path:{path}")
        if not path or path.startswith("real:"):
            unauditable.append(f"{_sample_id(entry)}:{source_id}:{path or '<empty>'}")
    return identities, unauditable


def _ids_from_rows(payload: dict[str, Any], artifact: str) -> tuple[list[str], list[str]]:
    rows = payload.get("samples")
    if not isinstance(rows, list):
        return [], [f"{artifact}.samples is missing or is not a list"]
    ids = [str(row.get("sample_id", "")) for row in rows if isinstance(row, dict)]
    errors: list[str] = []
    if any(not sample_id for sample_id in ids):
        errors.append(f"{artifact}.samples contains an empty sample_id")
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"{artifact}.samples contains duplicate IDs: {duplicates[:20]}")
    return ids, errors


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means = (
        "event_precision",
        "event_recall",
        "event_f1",
        "caption_token_f1",
        "seg_f1_100ms",
        "onset_mae",
        "offset_mae",
        "tolerance_acc_010",
        "source_count_mae",
        "pointer_accuracy",
    )
    totals = ("hallucination", "omission")
    result: dict[str, Any] = {"n_samples": len(rows)}
    for key in means:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        result[f"mean_{key}"] = round(sum(values) / len(values), 6) if values else None
    for key in totals:
        values = [int(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        result[f"total_{key}"] = sum(values) if values else None
    return result


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def audit_evaluation_result(
    *,
    train_config_path: str | Path,
    eval_manifest_path: str | Path,
    metrics_path: str | Path,
    inference_report_path: str | Path,
    repo_root: str | Path = ".",
    split_contract_path: str | Path | None = None,
    data_gate_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed, machine-readable validity report."""
    root = Path(repo_root).expanduser().resolve()
    config_path = _resolve_path(train_config_path, root)
    eval_path = _resolve_path(eval_manifest_path, root)
    metrics_file = _resolve_path(metrics_path, root)
    inference_file = _resolve_path(inference_report_path, root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    inference = json.loads(inference_file.read_text(encoding="utf-8"))
    train_manifest_path = _resolve_path(config["data"]["manifest_path"], root)
    train_manifest_entries = read_manifest(train_manifest_path)
    actual_train_entries = reconstruct_training_entries(config, train_manifest_entries)
    eval_entries = read_manifest(eval_path)

    train_ids = {_sample_id(entry) for entry in actual_train_entries}
    eval_manifest_ids = [_sample_id(entry) for entry in eval_entries]
    eval_manifest_id_set = set(eval_manifest_ids)
    metric_ids, metric_id_errors = _ids_from_rows(metrics, "metrics")
    inference_ids, inference_id_errors = _ids_from_rows(inference, "inference_report")
    reported_ids = set(metric_ids)
    reported_eval_entries = [
        entry for entry in eval_entries if _sample_id(entry) in reported_ids
    ]

    train_source_ids: set[str] = set()
    eval_source_ids: set[str] = set()
    unauditable_sources: list[str] = []
    for entry in actual_train_entries:
        identities, invalid = _source_identities(entry)
        train_source_ids.update(identities)
        unauditable_sources.extend(invalid)
    for entry in reported_eval_entries:
        identities, invalid = _source_identities(entry)
        eval_source_ids.update(identities)
        unauditable_sources.extend(invalid)

    sample_overlap = sorted(train_ids & reported_ids)
    source_overlap = sorted(train_source_ids & eval_source_ids)
    checks = [
        _check("training_membership_nonempty", bool(train_ids), len(train_ids)),
        _check(
            "evaluation_manifest_ids_unique",
            len(eval_manifest_ids) == len(eval_manifest_id_set),
            len(eval_manifest_ids) - len(eval_manifest_id_set),
        ),
        _check(
            "metrics_ids_well_formed",
            not metric_id_errors and metrics.get("n_samples") == len(metric_ids),
            {
                "errors": metric_id_errors,
                "declared": metrics.get("n_samples"),
                "observed": len(metric_ids),
            },
        ),
        _check(
            "inference_ids_well_formed",
            not inference_id_errors
            and inference.get("n_samples") == len(inference_ids),
            {
                "errors": inference_id_errors,
                "declared": inference.get("n_samples"),
                "observed": len(inference_ids),
            },
        ),
        _check(
            "metrics_inference_ids_identical",
            metric_ids == inference_ids,
            {
                "metrics_only": sorted(set(metric_ids) - set(inference_ids))[:20],
                "inference_only": sorted(set(inference_ids) - set(metric_ids))[:20],
                "same_order": metric_ids == inference_ids,
            },
        ),
        _check(
            "metrics_parser_evidence_identical",
            all(
                isinstance(metric_row, dict)
                and isinstance(inference_row, dict)
                and metric_row.get("strict_format_success")
                == inference_row.get("strict_format_success")
                for metric_row, inference_row in zip(
                    metrics.get("samples", []),
                    inference.get("samples", []),
                    strict=False,
                )
            )
            and metrics.get("strict_format_success_rate")
            == inference.get("strict_format_success_rate"),
            {
                "metrics_rate": metrics.get("strict_format_success_rate"),
                "inference_rate": inference.get("strict_format_success_rate"),
                "mismatched_sample_ids": [
                    str(metric_row.get("sample_id", ""))
                    for metric_row, inference_row in zip(
                        metrics.get("samples", []),
                        inference.get("samples", []),
                        strict=False,
                    )
                    if isinstance(metric_row, dict)
                    and isinstance(inference_row, dict)
                    and metric_row.get("strict_format_success")
                    != inference_row.get("strict_format_success")
                ][:20],
            },
        ),
        _check(
            "metrics_pointer_evidence_identical",
            all(
                isinstance(metric_row, dict)
                and isinstance(inference_row, dict)
                and metric_row.get("explicit_track_ids_complete")
                == inference_row.get("explicit_track_ids_complete", False)
                for metric_row, inference_row in zip(
                    metrics.get("samples", []),
                    inference.get("samples", []),
                    strict=False,
                )
            ),
            {
                "metrics_complete": metrics.get("pointer_evidence_complete"),
                "inference_complete_count": inference.get(
                    "n_explicit_track_ids_complete"
                ),
            },
        ),
        _check(
            "complete_evaluation_manifest_coverage",
            reported_ids == eval_manifest_id_set,
            {
                "missing_from_reports": sorted(eval_manifest_id_set - reported_ids)[:20],
                "unknown_report_ids": sorted(reported_ids - eval_manifest_id_set)[:20],
                "n_manifest": len(eval_manifest_id_set),
                "n_reported": len(reported_ids),
            },
        ),
        _check(
            "train_eval_sample_ids_disjoint",
            not sample_overlap,
            {"count": len(sample_overlap), "examples": sample_overlap[:20]},
        ),
        _check(
            "raw_source_identity_auditable",
            not unauditable_sources,
            {"count": len(unauditable_sources), "examples": unauditable_sources[:20]},
        ),
        _check(
            "train_eval_raw_sources_disjoint",
            not source_overlap,
            {"count": len(source_overlap), "examples": source_overlap[:20]},
        ),
    ]

    metric_schema_error: str | None = None
    metric_schema_summary: dict[str, Any] = {}
    try:
        metric_schema_summary = validate_metrics_artifact(metrics)
    except (TypeError, ValueError) as exc:
        metric_schema_error = str(exc)
    checks.append(
        _check(
            "current_semantic_metric_schema_valid",
            metric_schema_error is None,
            metric_schema_error or metric_schema_summary,
        )
    )

    contract_error: str | None = None
    contract = None
    contract_artifacts: dict[str, dict[str, str]] = {}
    if split_contract_path is None or data_gate_summary_path is None:
        contract_error = "--split-contract and --data-gate-summary are both required"
    else:
        contract_path = _resolve_path(split_contract_path, root)
        data_gate_path = _resolve_path(data_gate_summary_path, root)
        if contract_path.is_file():
            contract_artifacts["split_contract"] = {
                "path": _display_path(contract_path, root),
                "sha256": file_sha256(contract_path),
            }
        if data_gate_path.is_file():
            contract_artifacts["data_gate_summary"] = {
                "path": _display_path(data_gate_path, root),
                "sha256": file_sha256(data_gate_path),
            }
        try:
            contract = require_experiment_data_summary(data_gate_path, contract_path)
            require_split_manifest(contract_path, "train", train_manifest_path)
            require_split_manifest(contract_path, "test", eval_path)
            if config.get("data", {}).get("pre_split") is not True:
                raise ValueError("certified training requires data.pre_split=true")
            if config.get("data", {}).get("expected_split") != "train":
                raise ValueError("training config is not bound to the train split")
            if config.get("experiment_contract", {}).get("dataset_id") != contract.get(
                "dataset_id"
            ):
                raise ValueError("training config dataset_id does not match the data contract")
            if config.get("data", {}).get("require_human_audit") is not True:
                raise ValueError("certified training requires a passed human audit")
            human_path = _resolve_path(
                config.get("data", {}).get("human_audit_summary_path", ""), root
            )
            require_human_audit_summary(
                human_path, expected_dataset_id=contract["dataset_id"]
            )
            contract_artifacts["human_audit_summary"] = {
                "path": _display_path(human_path, root),
                "sha256": file_sha256(human_path),
            }
            if metrics.get("experiment_contract", {}).get("dataset_id") != contract.get(
                "dataset_id"
            ):
                raise ValueError("metrics dataset_id does not match the data contract")
            if metrics.get("experiment_contract", {}).get("split") != "test":
                raise ValueError("metrics are not bound to the test split")
            if inference.get("dataset_id") != contract.get("dataset_id"):
                raise ValueError("inference report dataset_id does not match the data contract")
            if inference.get("expected_split") != "test":
                raise ValueError("inference report is not bound to the test split")
            if not inference.get("prediction_sha256"):
                raise ValueError("inference report does not bind its predictions")
            metric_inference = metrics.get("inference_evidence", {})
            if metric_inference.get("sha256") != file_sha256(inference_file):
                raise ValueError("metrics do not bind this inference report")
            if metric_inference.get("prediction_sha256") != inference.get(
                "prediction_sha256"
            ):
                raise ValueError("metrics and inference report bind different predictions")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            contract_error = str(exc)
    checks.append(
        _check(
            "frozen_experiment_contract_valid",
            contract_error is None,
            contract_error or {"dataset_id": contract.get("dataset_id")},
        )
    )

    metric_rows = [row for row in metrics.get("samples", []) if isinstance(row, dict)]
    seen_rows = [row for row in metric_rows if str(row.get("sample_id")) in train_ids]
    unseen_rows = [row for row in metric_rows if str(row.get("sample_id")) not in train_ids]
    failed = [item["name"] for item in checks if item["pass"] is not True]
    return {
        "schema_version": "sceneledger-evaluation-validity-v1",
        "pass": not failed,
        "status": "certified_generalization" if not failed else "invalid_generalization_claim",
        "claim_scope": "paper_eligible" if not failed else "diagnostic_only",
        "dataset_id": contract.get("dataset_id") if contract_error is None else None,
        "failed_checks": failed,
        "artifacts": {
            "train_config": {
                "path": _display_path(config_path, root),
                "sha256": file_sha256(config_path),
            },
            "train_manifest": {
                "path": _display_path(train_manifest_path, root),
                "sha256": file_sha256(train_manifest_path),
            },
            "evaluation_manifest": {
                "path": _display_path(eval_path, root),
                "sha256": file_sha256(eval_path),
            },
            "metrics": {
                "path": _display_path(metrics_file, root),
                "sha256": file_sha256(metrics_file),
            },
            "inference_report": {
                "path": _display_path(inference_file, root),
                "sha256": file_sha256(inference_file),
            },
            **contract_artifacts,
        },
        "counts": {
            "train_manifest": len(train_manifest_entries),
            "actual_training": len(train_ids),
            "evaluation_manifest": len(eval_manifest_id_set),
            "reported_evaluation": len(reported_ids),
            "reported_seen_during_training": len(sample_overlap),
            "reported_unseen_by_sample_id": len(reported_ids - train_ids),
        },
        "checks": checks,
        "metric_subgroups": {
            "seen_during_training": _metric_summary(seen_rows),
            "unseen_by_sample_id": _metric_summary(unseen_rows),
        },
        "headline_metrics": {
            "event_f1": metrics.get("macro_event_f1"),
            "caption_token_f1": metrics.get("macro_caption_token_f1"),
            "seg_f1_100ms": metrics.get("macro_seg_f1_100ms"),
            "tolerance_acc_010": metrics.get("macro_tolerance_acc_010"),
            "source_count_mae": metrics.get("mean_source_count_mae"),
            "pointer_accuracy": metrics.get("mean_pointer_accuracy"),
            "pointer_metric": metrics.get("pointer_metric"),
            "pointer_evidence_complete": metrics.get(
                "pointer_evidence_complete"
            ),
            "explicit_track_ids_complete_rate": metrics.get(
                "explicit_track_ids_complete_rate"
            ),
            "hallucination": metrics.get("total_hallucination"),
            "omission": metrics.get("total_omission"),
        },
        "interpretation": (
            "Only a passing report supports a held-out generalization claim. "
            "The unseen-by-sample-ID subgroup is diagnostic when source identities "
            "are not auditable; it is not a source-disjoint test result."
        ),
    }


__all__ = ["audit_evaluation_result", "reconstruct_training_entries"]
