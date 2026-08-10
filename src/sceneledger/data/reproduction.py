"""Fail-closed acceptance checks for a frozen B3-valid data release."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sceneledger.data.datamodule import source_leakage
from sceneledger.data.manifests import file_hash, read_manifest

REQUIRED_SOURCE_KINDS = ("speech", "vocal", "music", "sfx", "ambience")


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _identity_hash(values: dict[str, str]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def require_b3_data_summary(
    path: str | Path, *, expected_dataset_id: str | None = None
) -> dict:
    """Load a passed data summary or raise before model work starts."""
    summary_path = Path(path).resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"B3 data summary missing: {summary_path}")
    payload = _load_json(summary_path)
    if payload.get("pass") is not True or payload.get("failed_checks"):
        raise ValueError(
            f"B3 data reproduction has not passed: {payload.get('failed_checks', [])}"
        )
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise ValueError("B3 data summary has no dataset_id")
    if expected_dataset_id and dataset_id != expected_dataset_id:
        raise ValueError(f"B3 dataset ID {dataset_id} != expected {expected_dataset_id}")
    return payload


def validate_b3_data_release(
    *,
    source_report_path: str | Path,
    source_readiness_report_path: str | Path,
    render_report_path: str | Path,
    sft_metadata_path: str | Path,
    train_manifest_path: str | Path,
    val_manifest_path: str | Path,
    expected_samples: int,
    allow_unknown_license: bool = False,
    git_commit: str | None = None,
) -> dict:
    """Return an auditable summary; every unmet requirement becomes a check."""
    if expected_samples <= 0:
        raise ValueError("expected_samples must be positive")

    paths = {
        "source_report": Path(source_report_path).resolve(),
        "source_readiness_report": Path(source_readiness_report_path).resolve(),
        "render_report": Path(render_report_path).resolve(),
        "sft_metadata": Path(sft_metadata_path).resolve(),
        "train_manifest": Path(train_manifest_path).resolve(),
        "val_manifest": Path(val_manifest_path).resolve(),
    }
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    documents: dict[str, dict] = {}
    for name in (
        "source_report",
        "source_readiness_report",
        "render_report",
        "sft_metadata",
    ):
        path = paths[name]
        check(f"{name}_exists", path.is_file(), str(path))
        if path.is_file():
            try:
                documents[name] = _load_json(path)
                check(f"{name}_valid_json", True, "JSON object")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                check(f"{name}_valid_json", False, str(exc))

    source = documents.get("source_report", {})
    if source:
        kinds = source.get("kinds", {})
        missing_kinds = [
            kind
            for kind in REQUIRED_SOURCE_KINDS
            if _safe_int(kinds.get(kind, 0), 0) <= 0
        ]
        check("all_source_kinds_present", not missing_kinds, missing_kinds)
        check(
            "source_audio_files_verified",
            source.get("missing_files_allowed") is False
            and source.get("all_files_verified") is True,
            {
                "missing_files_allowed": source.get("missing_files_allowed"),
                "all_files_verified": source.get("all_files_verified"),
            },
        )
        vocal_count = _safe_int(kinds.get("vocal", 0), 0)
        verbatim_count = _safe_int(source.get("vocal_with_verbatim_lyrics", 0), 0)
        check(
            "all_vocals_have_verbatim_lyrics",
            vocal_count > 0 and verbatim_count == vocal_count,
            {"vocal": vocal_count, "verbatim": verbatim_count},
        )
        licenses = source.get("licenses", {})
        unknown_licenses = _safe_int(
            licenses.get("unknown", 0) if isinstance(licenses, dict) else None
        )
        check(
            "source_licenses_known",
            allow_unknown_license or unknown_licenses == 0,
            {"unknown": unknown_licenses, "allowed": allow_unknown_license},
        )
        output_path = Path(str(source.get("output", "")))
        output_exists = output_path.is_file()
        check("canonical_source_catalog_exists", output_exists, str(output_path))
        if output_exists:
            actual_hash = file_hash(output_path)
            check(
                "canonical_source_catalog_hash_matches",
                source.get("output_sha256") == actual_hash,
                {"reported": source.get("output_sha256"), "actual": actual_hash},
            )

    readiness = documents.get("source_readiness_report", {})
    if readiness:
        check(
            "source_readiness_passed",
            readiness.get("pass") is True and not readiness.get("failed_checks"),
            readiness.get("failed_checks", []),
        )
        check(
            "source_pool_id_present",
            bool(readiness.get("source_pool_id")),
            readiness.get("source_pool_id"),
        )
        check(
            "all_source_audio_audited",
            _safe_int(readiness.get("n_audio_ok"))
            == _safe_int(readiness.get("n_sources"))
            and _safe_int(readiness.get("n_sources")) > 0,
            {
                "n_audio_ok": readiness.get("n_audio_ok"),
                "n_sources": readiness.get("n_sources"),
            },
        )
        inventory_path = Path(str(readiness.get("inventory_path", "")))
        inventory_exists = inventory_path.is_file()
        check("source_inventory_exists", inventory_exists, str(inventory_path))
        if inventory_exists:
            actual_hash = file_hash(inventory_path)
            check(
                "source_inventory_hash_matches",
                readiness.get("inventory_sha256") == actual_hash,
                {"reported": readiness.get("inventory_sha256"), "actual": actual_hash},
            )
        if source:
            check(
                "readiness_uses_canonical_source_catalog",
                readiness.get("source_catalog_sha256") == source.get("output_sha256"),
                {
                    "readiness": readiness.get("source_catalog_sha256"),
                    "source": source.get("output_sha256"),
                },
            )

    render = documents.get("render_report", {})
    if render:
        check("render_validation_passed", render.get("pass") is True, render.get("failures", []))
        n_entries = _safe_int(render.get("n_entries"))
        check(
            "rendered_sample_count",
            n_entries == expected_samples,
            {"expected": expected_samples, "actual": n_entries},
        )
        for field in (
            "n_replay_ok",
            "n_stems_sum_ok",
            "n_ledger_valid",
            "n_saved_reconstruction_ok",
        ):
            actual = _safe_int(render.get(field))
            check(
                f"{field}_complete",
                actual == expected_samples,
                {"expected": expected_samples, "actual": actual},
            )
        for field in (
            "n_replay_fail",
            "n_stems_sum_fail",
            "n_ledger_invalid",
            "n_audio_files_fail",
            "n_saved_reconstruction_fail",
        ):
            actual = _safe_int(render.get(field))
            check(f"{field}_zero", actual == 0, actual)
        manifest_path = Path(str(render.get("manifest_path", "")))
        manifest_exists = manifest_path.is_file()
        check("render_manifest_exists", manifest_exists, str(manifest_path))
        if manifest_exists:
            actual_hash = file_hash(manifest_path)
            check(
                "render_manifest_hash_matches",
                render.get("manifest_sha256") == actual_hash,
                {"reported": render.get("manifest_sha256"), "actual": actual_hash},
            )
        if source:
            check(
                "render_uses_canonical_source_catalog",
                render.get("source_catalog_sha256") == source.get("output_sha256"),
                {
                    "render": render.get("source_catalog_sha256"),
                    "source": source.get("output_sha256"),
                },
            )

    metadata = documents.get("sft_metadata", {})
    if metadata:
        check(
            "sft_total_matches_expected",
            _safe_int(metadata.get("n_total")) == expected_samples,
            {"expected": expected_samples, "actual": metadata.get("n_total")},
        )
        check("sft_structural_audit", metadata.get("structural_audit_ok") is True, metadata.get("structural_audit_errors", []))
        check("sft_source_leakage_zero", _safe_int(metadata.get("source_leakage_count")) == 0, metadata.get("source_leakage_count"))
        check("sft_missing_audio_zero", _safe_int(metadata.get("missing_audio_count")) == 0, metadata.get("missing_audio_count"))
        check("sft_placeholder_lyrics_zero", _safe_int(metadata.get("placeholder_lyrics_count")) == 0, metadata.get("placeholder_lyrics_count"))
        check("sft_tracks_enabled", metadata.get("include_tracks") is True, metadata.get("include_tracks"))
        check("sft_lyrics_enabled", metadata.get("include_lyrics") is True, metadata.get("include_lyrics"))
        check("sft_target_is_atomic", metadata.get("target_mode") == "atomic", metadata.get("target_mode"))
        if render:
            check(
                "sft_uses_validated_render_manifest",
                metadata.get("manifest_sha256") == render.get("manifest_sha256"),
                {
                    "sft": metadata.get("manifest_sha256"),
                    "render": render.get("manifest_sha256"),
                },
            )

    train_entries = []
    val_entries = []
    for name in ("train_manifest", "val_manifest"):
        path = paths[name]
        check(f"{name}_exists", path.is_file(), str(path))
        if path.is_file():
            try:
                entries = read_manifest(path)
                if name == "train_manifest":
                    train_entries = entries
                else:
                    val_entries = entries
                check(f"{name}_readable", True, len(entries))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                check(f"{name}_readable", False, str(exc))

    if train_entries and val_entries:
        train_ids = {str(entry.scene["scene_id"]) for entry in train_entries}
        val_ids = {str(entry.scene["scene_id"]) for entry in val_entries}
        leaked_sources = source_leakage(train_entries, val_entries)
        check("train_val_sample_ids_disjoint", not train_ids & val_ids, sorted(train_ids & val_ids)[:20])
        check("train_val_sources_disjoint", not leaked_sources, sorted(leaked_sources)[:20])
        check(
            "frozen_split_covers_dataset",
            len(train_entries) + len(val_entries) == expected_samples,
            {"train": len(train_entries), "val": len(val_entries)},
        )
        if metadata:
            check("train_count_matches_metadata", len(train_entries) == _safe_int(metadata.get("n_train")), metadata.get("n_train"))
            check("val_count_matches_metadata", len(val_entries) == _safe_int(metadata.get("n_val")), metadata.get("n_val"))
            for name, path in (
                ("train", paths["train_manifest"]),
                ("val", paths["val_manifest"]),
            ):
                actual_hash = file_hash(path)
                check(
                    f"{name}_manifest_hash_matches_metadata",
                    metadata.get(f"{name}_manifest_sha256") == actual_hash,
                    {
                        "reported": metadata.get(f"{name}_manifest_sha256"),
                        "actual": actual_hash,
                    },
                )

    artifact_hashes = {
        name: file_hash(path) for name, path in paths.items() if path.is_file()
    }
    if render:
        render_manifest = Path(str(render.get("manifest_path", "")))
        if render_manifest.is_file():
            artifact_hashes["render_manifest"] = file_hash(render_manifest)
    if source:
        source_catalog = Path(str(source.get("output", "")))
        if source_catalog.is_file():
            artifact_hashes["source_catalog"] = file_hash(source_catalog)
    if readiness:
        source_inventory = Path(str(readiness.get("inventory_path", "")))
        if source_inventory.is_file():
            artifact_hashes["source_inventory"] = file_hash(source_inventory)

    identity_hashes = {
        key: artifact_hashes[key]
        for key in (
            "source_catalog",
            "source_inventory",
            "render_manifest",
            "train_manifest",
            "val_manifest",
        )
        if key in artifact_hashes
    }
    passed = bool(checks) and all(item["pass"] is True for item in checks)
    return {
        "schema_version": "b3-data-reproduction-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": passed,
        "status": "passed" if passed else "failed",
        "git_commit": git_commit,
        "expected_samples": expected_samples,
        "n_train": len(train_entries),
        "n_val": len(val_entries),
        "dataset_id": (
            _identity_hash(identity_hashes) if len(identity_hashes) == 5 else None
        ),
        "source_pool_id": readiness.get("source_pool_id") if readiness else None,
        "artifact_hashes": artifact_hashes,
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if item["pass"] is not True],
    }


__all__ = [
    "REQUIRED_SOURCE_KINDS",
    "require_b3_data_summary",
    "validate_b3_data_release",
]
