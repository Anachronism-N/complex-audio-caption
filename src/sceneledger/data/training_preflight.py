"""CPU-only authorization gate executed before a training model is loaded."""

from __future__ import annotations

import hashlib
import json
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
from sceneledger.eval.result_validity import reconstruct_training_entries

PREFLIGHT_SCHEMA_VERSION = "sceneledger-training-preflight-v1"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def _source_diagnostics(entries: list[ManifestEntry]) -> dict[str, Any]:
    placeholder_sources: list[str] = []
    source_groups: set[str] = set()
    source_paths: set[str] = set()
    source_count = 0
    complete_waveform_hashes = 0
    complete_stem_evidence = 0
    for entry in entries:
        sample_id = str(entry.scene.get("scene_id", ""))
        source_ids = {
            str(source.get("source_id", ""))
            for source in entry.scene.get("sources", [])
            if source.get("source_id") not in (None, "")
        }
        if entry.mixture_hash and entry.dry_mixture_hash:
            complete_waveform_hashes += 1
        if (
            source_ids
            and source_ids <= set(entry.stem_paths)
            and source_ids <= set(entry.stem_hashes)
            and source_ids <= set(entry.activity_hashes)
        ):
            complete_stem_evidence += 1
        for source in entry.scene.get("sources", []):
            source_count += 1
            path = str(source.get("path", "")).replace("\\", "/")
            group = source.get("source_group")
            leakage_groups = source.get("leakage_groups", [])
            if path:
                source_paths.add(path)
            if group not in (None, ""):
                source_groups.add(str(group))
            source_groups.update(
                str(value) for value in leakage_groups if value not in (None, "")
            )
            if (not path or path.startswith("real:")) and group in (None, ""):
                placeholder_sources.append(
                    f"{sample_id}:{source.get('source_id', 'unknown')}:{path or '<empty>'}"
                )
    return {
        "n_sources": source_count,
        "n_source_groups": len(source_groups),
        "n_unique_paths": len(source_paths),
        "unique_paths_preview": sorted(source_paths)[:20],
        "n_placeholder_sources": len(placeholder_sources),
        "placeholder_examples": placeholder_sources[:20],
        "n_nonempty_mixture_hashes": sum(bool(entry.mixture_hash) for entry in entries),
        "n_nonempty_stem_maps": sum(bool(entry.stem_paths) for entry in entries),
        "n_complete_waveform_hashes": complete_waveform_hashes,
        "n_complete_stem_evidence": complete_stem_evidence,
    }


def audit_training_config(
    config_path: str | Path,
    *,
    repo_root: str | Path = ".",
    allow_exploratory_uncontracted: bool = False,
) -> dict[str, Any]:
    """Authorize contracted training or explicitly label exploratory training."""
    root = Path(repo_root).expanduser().resolve()
    config_file = _resolve(config_path, root)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("training config must contain a data mapping")
    data = config["data"]
    manifest_value = data.get("manifest_path")
    if not manifest_value:
        raise ValueError("training config is missing data.manifest_path")
    manifest_path = _resolve(str(manifest_value), root)
    entries = read_manifest(manifest_path)
    actual_training = reconstruct_training_entries(config, entries)
    diagnostics = _source_diagnostics(entries)
    sample_ids = [str(entry.scene.get("scene_id", "")) for entry in entries]
    actual_training_ids = {
        str(entry.scene.get("scene_id", "")) for entry in actual_training
    }
    internal_validation = [
        entry
        for entry in entries
        if str(entry.scene.get("scene_id", "")) not in actual_training_ids
    ]

    required_contract_fields = {
        "pre_split": data.get("pre_split") is True,
        "expected_split_train": data.get("expected_split") == "train",
        "split_contract_path": bool(data.get("split_contract_path")),
        "data_gate_summary_path": bool(data.get("data_gate_summary_path")),
        "require_human_audit": data.get("require_human_audit") is True,
        "human_audit_summary_path": bool(data.get("human_audit_summary_path")),
    }
    any_contract_field = any(
        (
            data.get("pre_split") is not None,
            bool(data.get("expected_split")),
            bool(data.get("split_contract_path")),
            bool(data.get("data_gate_summary_path")),
            bool(data.get("human_audit_summary_path")),
        )
    )
    complete_contract_shape = all(required_contract_fields.values())
    checks = [
        _check("manifest_nonempty", bool(entries), len(entries)),
        _check("training_membership_nonempty", bool(actual_training), len(actual_training)),
        _check(
            "sample_ids_well_formed",
            all(sample_ids) and len(set(sample_ids)) == len(sample_ids),
            {
                "n_samples": len(sample_ids),
                "n_empty": sum(not sample_id for sample_id in sample_ids),
                "n_unique": len(set(sample_ids)),
            },
        ),
        _check(
            "raw_source_identity_auditable",
            diagnostics["n_placeholder_sources"] == 0,
            {
                "n_placeholder_sources": diagnostics["n_placeholder_sources"],
                "examples": diagnostics["placeholder_examples"],
            },
        ),
        _check(
            "mixture_hashes_complete",
            diagnostics["n_complete_waveform_hashes"] == len(entries),
            f"{diagnostics['n_complete_waveform_hashes']}/{len(entries)}",
        ),
        _check(
            "stem_evidence_complete",
            diagnostics["n_complete_stem_evidence"] == len(entries),
            f"{diagnostics['n_complete_stem_evidence']}/{len(entries)}",
        ),
    ]
    dataset_id: str | None = None
    contract_error: str | None = None
    contract_valid = False
    contract_artifacts: dict[str, Any] = {}
    publication_eligible = False

    if complete_contract_shape:
        split_contract = _resolve(str(data["split_contract_path"]), root)
        data_summary = _resolve(str(data["data_gate_summary_path"]), root)
        human_summary = _resolve(str(data["human_audit_summary_path"]), root)
        try:
            gate = require_experiment_data_summary(data_summary, split_contract)
            contract = require_split_manifest(split_contract, "train", manifest_path)
            if gate.get("dataset_id") != contract.get("dataset_id"):
                raise ValueError("data gate and split contract dataset IDs differ")
            require_human_audit_summary(
                human_summary, expected_dataset_id=str(gate["dataset_id"])
            )
            dataset_id = str(gate["dataset_id"])
            if config.get("experiment_contract", {}).get("dataset_id") != dataset_id:
                raise ValueError("training config dataset_id differs from data contract")
            contract_valid = True
            contract_artifacts = {
                "split_contract": {
                    "path": _display_path(split_contract, root),
                    "sha256": file_sha256(split_contract),
                },
                "data_gate_summary": {
                    "path": _display_path(data_summary, root),
                    "sha256": file_sha256(data_summary),
                },
                "human_audit_summary": {
                    "path": _display_path(human_summary, root),
                    "sha256": file_sha256(human_summary),
                },
            }
            publication_eligible = contract_valid and all(
                check["pass"] is True for check in checks
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            contract_error = str(exc)
        checks.append(
            _check(
                "frozen_training_contract_valid",
                contract_valid,
                contract_error or {"dataset_id": dataset_id},
            )
        )
    else:
        missing = [name for name, present in required_contract_fields.items() if not present]
        checks.append(
            _check(
                "frozen_training_contract_valid",
                False,
                {
                    "missing": missing,
                    "partial_contract": any_contract_field,
                },
            )
        )

    if publication_eligible:
        authorized = True
        status = "contracted_paper_eligible"
    elif allow_exploratory_uncontracted and not any_contract_field:
        authorized = True
        status = "exploratory_uncontracted"
    else:
        authorized = False
        status = "rejected_uncontracted_or_invalid"
    checks.append(
        _check(
            "training_authorized",
            authorized,
            {
                "allow_exploratory_uncontracted": allow_exploratory_uncontracted,
                "status": status,
            },
        )
    )

    publication_blockers: list[str] = []
    if not contract_valid:
        publication_blockers.append("no_valid_frozen_training_contract")
    if diagnostics["n_placeholder_sources"]:
        publication_blockers.append("raw_source_identity_not_auditable")
    if diagnostics["n_complete_waveform_hashes"] != len(entries):
        publication_blockers.append("mixture_hashes_incomplete")
    if diagnostics["n_complete_stem_evidence"] != len(entries):
        publication_blockers.append("stem_evidence_incomplete")
    if data.get("pre_split") is not True:
        publication_blockers.append("temporary_in_trainer_split")

    identity = {
        "config_sha256": file_sha256(config_file),
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_id": dataset_id,
        "status": status,
        "actual_training_sample_ids": sorted(
            str(entry.scene.get("scene_id", "")) for entry in actual_training
        ),
    }
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "preflight_id": _canonical_hash(identity),
        "pass": authorized,
        "authorized_to_train": authorized,
        "publication_eligible": publication_eligible,
        "status": status,
        "dataset_id": dataset_id,
        "config": {
            "path": _display_path(config_file, root),
            "sha256": identity["config_sha256"],
        },
        "manifest": {
            "path": _display_path(manifest_path, root),
            "sha256": identity["manifest_sha256"],
            "n_samples": len(entries),
            "n_actual_training_samples": len(actual_training),
            "n_internal_validation_samples": len(entries) - len(actual_training),
        },
        "split_diagnostics": {
            "actual_training_template_counts": dict(
                sorted(
                    Counter(
                        str(entry.scene.get("template", "unknown"))
                        for entry in actual_training
                    ).items()
                )
            ),
            "internal_validation_template_counts": dict(
                sorted(
                    Counter(
                        str(entry.scene.get("template", "unknown"))
                        for entry in internal_validation
                    ).items()
                )
            ),
        },
        "source_diagnostics": diagnostics,
        "publication_blockers": publication_blockers,
        "contract_artifacts": contract_artifacts,
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if check["pass"] is not True],
    }


def write_training_preflight(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


__all__ = ["audit_training_config", "write_training_preflight"]
