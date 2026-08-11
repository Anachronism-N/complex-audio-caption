"""Fail-closed contracts for paper-valid SceneLedger experiments.

This module deliberately separates two questions that were previously mixed:

1. Are train/validation/test identities frozen and mutually isolated?
2. Does each rendered fold resemble the intended complex-audio distribution?

The functions are CPU-only and operate on manifests, so they can run before a
GPU job is submitted.  A failed contract is an expected result for invalid
data, not a reason to continue training.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import yaml

from sceneledger.data.manifests import ManifestEntry, read_manifest
from sceneledger.data.schema import Ledger

SPLIT_NAMES = ("train", "val", "test")


def file_sha256(path: str | Path) -> str:
    """Return the complete SHA-256 identity of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_sha256(values: object) -> str:
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_id(entry: ManifestEntry) -> str:
    return str(entry.scene["scene_id"])


def _source_identities(entry: ManifestEntry) -> set[str]:
    """Return raw-recording identities used by a rendered scene.

    ``source_group`` is preferred because multiple segments from one original
    recording must remain in the same fold.  Older manifests fall back to the
    source path.  Generated per-scene ``source_id`` values are intentionally
    not used: they do not identify the underlying dry recording.
    """
    identities: set[str] = set()
    for source in entry.scene.get("sources", []):
        group = source.get("source_group")
        if group not in (None, ""):
            identities.add(f"group:{group}")
            continue
        path = str(source.get("path", "")).replace("\\", "/")
        if path:
            identities.add(f"path:{path}")
    return identities


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _check(name: str, passed: bool, detail: object) -> dict[str, object]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def build_split_contract(
    *,
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path,
    seed: int | None = None,
) -> dict:
    """Freeze three explicit manifests and reject sample/source leakage."""
    manifest_paths = {
        "train": Path(train_manifest).resolve(),
        "val": Path(val_manifest).resolve(),
        "test": Path(test_manifest).resolve(),
    }
    entries = {name: read_manifest(path) for name, path in manifest_paths.items()}
    checks: list[dict[str, object]] = []
    split_payload: dict[str, dict] = {}

    sample_sets: dict[str, set[str]] = {}
    source_sets: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        ids = [_sample_id(entry) for entry in entries[name]]
        duplicate_ids = _duplicates(ids)
        sources = {
            source
            for entry in entries[name]
            for source in _source_identities(entry)
        }
        sample_sets[name] = set(ids)
        source_sets[name] = sources
        checks.append(_check(f"{name}_nonempty", bool(ids), len(ids)))
        checks.append(
            _check(f"{name}_sample_ids_unique", not duplicate_ids, duplicate_ids[:20])
        )
        split_payload[name] = {
            "manifest_path": str(manifest_paths[name]),
            "manifest_sha256": file_sha256(manifest_paths[name]),
            "n_samples": len(ids),
            "n_source_identities": len(sources),
            "sample_ids_sha256": _identity_sha256(sorted(ids)),
            "sample_ids": sorted(ids),
        }

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        sample_overlap = sorted(sample_sets[left] & sample_sets[right])
        source_overlap = sorted(source_sets[left] & source_sets[right])
        checks.append(
            _check(
                f"{left}_{right}_sample_ids_disjoint",
                not sample_overlap,
                sample_overlap[:20],
            )
        )
        checks.append(
            _check(
                f"{left}_{right}_sources_disjoint",
                not source_overlap,
                source_overlap[:20],
            )
        )

    passed = all(item["pass"] is True for item in checks)
    identity = {
        name: {
            "manifest_sha256": split_payload[name]["manifest_sha256"],
            "sample_ids_sha256": split_payload[name]["sample_ids_sha256"],
        }
        for name in SPLIT_NAMES
    }
    return {
        "schema_version": "sceneledger-split-contract-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": passed,
        "status": "passed" if passed else "failed",
        "seed": seed,
        "dataset_id": _identity_sha256(identity),
        "splits": split_payload,
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if item["pass"] is not True],
    }


def write_split_contract(path: str | Path, contract: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_contract(path: str | Path) -> dict:
    contract_path = Path(path).resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"split contract must be a JSON object: {contract_path}")
    if payload.get("schema_version") != "sceneledger-split-contract-v1":
        raise ValueError(f"unsupported split contract schema: {payload.get('schema_version')}")
    if payload.get("pass") is not True or payload.get("failed_checks"):
        raise ValueError(
            f"split contract has not passed: {payload.get('failed_checks', [])}"
        )
    return payload


def require_split_manifest(
    contract_path: str | Path, split: str, manifest_path: str | Path
) -> dict:
    """Require that ``manifest_path`` is exactly the frozen requested fold."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    payload = _load_contract(contract_path)
    expected = payload.get("splits", {}).get(split)
    if not isinstance(expected, dict):
        raise ValueError(f"split {split!r} missing from contract")
    actual_hash = file_sha256(manifest_path)
    if actual_hash != expected.get("manifest_sha256"):
        raise ValueError(
            f"{split} manifest hash mismatch: {actual_hash} != "
            f"{expected.get('manifest_sha256')}"
        )
    entries = read_manifest(manifest_path)
    ids = sorted(_sample_id(entry) for entry in entries)
    if _identity_sha256(ids) != expected.get("sample_ids_sha256"):
        raise ValueError(f"{split} manifest sample IDs do not match split contract")
    return payload


def require_experiment_data_summary(
    summary_path: str | Path, split_contract_path: str | Path
) -> dict:
    """Require the complete split + quality gate and verify its artifact hashes."""
    path = Path(summary_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sceneledger-experiment-data-gate-v1":
        raise ValueError(f"unsupported experiment data summary schema in {path}")
    if payload.get("pass") is not True or payload.get("failed_checks"):
        raise ValueError(
            f"experiment data gate has not passed: {payload.get('failed_checks', [])}"
        )

    contract = _load_contract(split_contract_path)
    actual_contract_hash = file_sha256(split_contract_path)
    if payload.get("split_contract_sha256") != actual_contract_hash:
        raise ValueError("split contract hash does not match experiment data summary")
    if payload.get("dataset_id") != contract.get("dataset_id"):
        raise ValueError("dataset ID differs between data summary and split contract")

    quality_config = Path(str(payload.get("quality_config_path", "")))
    if not quality_config.is_file():
        raise ValueError(f"quality config missing: {quality_config}")
    if file_sha256(quality_config) != payload.get("quality_config_sha256"):
        raise ValueError("quality config changed after data gate was created")

    reports = payload.get("quality_reports", {})
    references = payload.get("references", {})
    for split in SPLIT_NAMES:
        item = reports.get(split)
        if not isinstance(item, dict) or item.get("pass") is not True:
            raise ValueError(f"quality report for {split} is missing or failed")
        report_path = Path(str(item.get("path", "")))
        if not report_path.is_file() or file_sha256(report_path) != item.get("sha256"):
            raise ValueError(f"quality report for {split} changed after data gate")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("pass") is not True or report.get("failed_checks"):
            raise ValueError(f"quality report for {split} has not passed")
        if (
            report.get("manifest_sha256")
            != contract["splits"][split]["manifest_sha256"]
        ):
            raise ValueError(f"quality report for {split} audits a different manifest")
        reference = references.get(split)
        if not isinstance(reference, dict):
            raise ValueError(f"frozen references for {split} are missing")
        reference_path = Path(str(reference.get("path", "")))
        if (
            not reference_path.is_file()
            or file_sha256(reference_path) != reference.get("sha256")
        ):
            raise ValueError(f"frozen references for {split} changed after data gate")
        reference_ids = sorted(_ledger_sample_ids(reference_path))
        if reference_ids != contract["splits"][split]["sample_ids"]:
            raise ValueError(f"frozen references for {split} have incorrect sample IDs")
    return payload


def _ledger_sample_ids(path: str | Path) -> list[str]:
    ids: list[str] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id")
        if not sample_id:
            raise ValueError(f"missing sample_id at {path}:{line_number}")
        ids.append(str(sample_id))
    duplicate_ids = _duplicates(ids)
    if duplicate_ids:
        raise ValueError(f"duplicate sample IDs in {path}: {duplicate_ids[:20]}")
    return ids


def require_ledger_split(
    contract_path: str | Path,
    split: str,
    ledger_path: str | Path,
    *,
    role: str,
) -> dict:
    """Require prediction/reference IDs to exactly equal a frozen split."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    payload = _load_contract(contract_path)
    expected = payload["splits"][split]
    actual_ids = sorted(_ledger_sample_ids(ledger_path))
    expected_ids = expected.get("sample_ids", [])
    if actual_ids != expected_ids:
        actual_set = set(actual_ids)
        expected_set = set(expected_ids)
        raise ValueError(
            f"{role} IDs do not match frozen {split} split: "
            f"missing={sorted(expected_set - actual_set)[:20]} "
            f"extra={sorted(actual_set - expected_set)[:20]}"
        )
    return payload


def write_references(manifest_path: str | Path, output_path: str | Path) -> int:
    """Export canonical Ledger references from a frozen manifest."""
    entries = read_manifest(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.target_ledger, ensure_ascii=False) + "\n")
    return len(entries)


def load_quality_profile(path: str | Path, profile: str) -> tuple[dict, str]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"quality profile {profile!r} not found in {config_path}")
    selected = profiles[profile]
    if not isinstance(selected, dict):
        raise ValueError(f"quality profile {profile!r} must be a mapping")
    return selected, file_sha256(config_path)


def _merged_spans(entry: ManifestEntry) -> list[tuple[float, float]]:
    duration = float(entry.scene["duration"])
    spans: list[tuple[float, float]] = []
    for event in entry.target_ledger.get("events", []):
        for span in event.get("spans", []):
            start = max(0.0, min(duration, float(span["start_sec"])))
            end = max(start, min(duration, float(span["end_sec"])))
            if end > start:
                spans.append((start, end))
    spans.sort()
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _track_overlap_ratio(entry: ManifestEntry) -> float:
    duration = float(entry.scene["duration"])
    if duration <= 0:
        return 0.0
    changes: defaultdict[float, int] = defaultdict(int)
    for track in entry.target_ledger.get("tracks", []):
        for span in track.get("spans", []):
            start = max(0.0, min(duration, float(span["start_sec"])))
            end = max(start, min(duration, float(span["end_sec"])))
            if end > start:
                changes[start] += 1
                changes[end] -= 1
    active = 0
    previous = 0.0
    overlap = 0.0
    for position in sorted(changes):
        if active >= 2:
            overlap += position - previous
        active += changes[position]
        previous = position
    return overlap / duration


def _scene_statistics(entry: ManifestEntry) -> dict:
    duration = float(entry.scene["duration"])
    merged = _merged_spans(entry)
    active_duration = sum(end - start for start, end in merged)
    previous = 0.0
    silences: list[float] = []
    for start, end in merged:
        silences.append(max(0.0, start - previous))
        previous = max(previous, end)
    trailing_silence = max(0.0, duration - previous)
    silences.append(trailing_silence)
    events = entry.target_ledger.get("events", [])
    source_ids = [str(source.get("source_id", "")) for source in entry.scene.get("sources", [])]
    sfx_span_counts = [
        len(event.get("spans", [])) for event in events if event.get("type") == "sfx"
    ]
    ledger_overlap = (
        entry.target_ledger.get("conditions", {}).get("overlap_ratio")
        if isinstance(entry.target_ledger.get("conditions"), dict)
        else None
    )
    overlap_ratio = (
        float(ledger_overlap) if ledger_overlap is not None else _track_overlap_ratio(entry)
    )
    return {
        "sample_id": _sample_id(entry),
        "template": str(entry.scene.get("template", "unknown")),
        "duration_sec": duration,
        "event_count": len(events),
        "span_count": sum(len(event.get("spans", [])) for event in events),
        "source_count": len(entry.scene.get("sources", [])),
        "active_ratio": active_duration / duration if duration > 0 else 0.0,
        "trailing_silence_sec": trailing_silence,
        "max_silence_sec": max(silences, default=duration),
        "overlap_ratio": overlap_ratio,
        "max_sfx_spans": max(sfx_span_counts, default=0),
        "duplicate_source_ids": len(source_ids) != len(set(source_ids)),
    }


def _fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def audit_mixture_distribution(
    manifest_path: str | Path,
    *,
    profile_name: str,
    profile: dict,
    config_sha256: str | None = None,
) -> dict:
    """Audit temporal density and complexity using canonical ledger spans."""
    entries = read_manifest(manifest_path)
    rows: list[dict] = []
    ledger_errors: list[str] = []
    for entry in entries:
        sample_id = _sample_id(entry)
        try:
            ledger = Ledger.model_validate(entry.target_ledger)
            if ledger.sample_id != sample_id:
                raise ValueError(
                    f"ledger sample_id {ledger.sample_id!r} != scene_id {sample_id!r}"
                )
            if abs(ledger.duration_sec - float(entry.scene["duration"])) > 1e-6:
                raise ValueError("ledger duration differs from scene duration")
        except Exception as exc:
            if len(ledger_errors) < 100:
                ledger_errors.append(f"{sample_id}: {exc}")
        try:
            rows.append(_scene_statistics(entry))
        except Exception as exc:
            if len(ledger_errors) < 100:
                ledger_errors.append(f"{sample_id}: statistics failed: {exc}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "template": str(entry.scene.get("template", "unknown")),
                    "duration_sec": float(entry.scene.get("duration", 0.0)),
                    "event_count": 0,
                    "span_count": 0,
                    "source_count": len(entry.scene.get("sources", [])),
                    "active_ratio": 0.0,
                    "trailing_silence_sec": float(entry.scene.get("duration", 0.0)),
                    "max_silence_sec": float(entry.scene.get("duration", 0.0)),
                    "overlap_ratio": 0.0,
                    "max_sfx_spans": 0,
                    "duplicate_source_ids": False,
                }
            )
    total = len(rows)
    global_cfg = profile.get("global", {})

    min_active = float(global_cfg.get("min_active_ratio", 0.3))
    tail_threshold = float(global_cfg.get("long_trailing_silence_sec", 5.0))
    silence_threshold = float(global_cfg.get("long_silence_sec", 5.0))
    single_count = sum(row["event_count"] <= 1 for row in rows)
    low_active_count = sum(row["active_ratio"] < min_active for row in rows)
    long_tail_count = sum(row["trailing_silence_sec"] > tail_threshold for row in rows)
    long_silence_count = sum(row["max_silence_sec"] > silence_threshold for row in rows)
    duplicate_source_count = sum(row["duplicate_source_ids"] for row in rows)

    sparse_cfg = profile.get("sparse_templates", {})
    sparse_names = set(sparse_cfg.get("names", ["isolated_sfx"]))
    sparse_count = sum(row["template"] in sparse_names for row in rows)

    repeated_cfg = profile.get("repeated_event", {})
    repeated_name = str(repeated_cfg.get("template", "repeated_event"))
    repeated_rows = [row for row in rows if row["template"] == repeated_name]
    min_repeated_spans = int(repeated_cfg.get("min_sfx_spans", 2))
    repeated_bad = sum(row["max_sfx_spans"] < min_repeated_spans for row in repeated_rows)

    overlap_cfg = profile.get("overlapping_speakers", {})
    overlap_name = str(overlap_cfg.get("template", "overlapping_speakers"))
    overlap_rows = [row for row in rows if row["template"] == overlap_name]
    min_overlap = float(overlap_cfg.get("min_overlap_ratio", 0.1))
    overlap_bad = sum(row["overlap_ratio"] < min_overlap for row in overlap_rows)

    metrics = {
        "n_samples": total,
        "single_event_fraction": _fraction(single_count, total),
        "low_active_fraction": _fraction(low_active_count, total),
        "long_trailing_silence_fraction": _fraction(long_tail_count, total),
        "long_silence_fraction": _fraction(long_silence_count, total),
        "duplicate_source_id_fraction": _fraction(duplicate_source_count, total),
        "sparse_template_fraction": _fraction(sparse_count, total),
        "repeated_event_violation_fraction": _fraction(repeated_bad, len(repeated_rows)),
        "overlap_violation_fraction": _fraction(overlap_bad, len(overlap_rows)),
        "mean_event_count": _mean(float(row["event_count"]) for row in rows),
        "mean_active_ratio": _mean(float(row["active_ratio"]) for row in rows),
        "mean_trailing_silence_sec": _mean(
            float(row["trailing_silence_sec"]) for row in rows
        ),
    }

    checks = [
        _check("manifest_nonempty", total > 0, total),
        _check("all_ledgers_schema_valid", not ledger_errors, ledger_errors),
        _check(
            "single_event_fraction",
            metrics["single_event_fraction"]
            <= float(global_cfg.get("max_single_event_fraction", 0.05)),
            {
                "actual": metrics["single_event_fraction"],
                "maximum": float(global_cfg.get("max_single_event_fraction", 0.05)),
            },
        ),
        _check(
            "low_active_fraction",
            metrics["low_active_fraction"]
            <= float(global_cfg.get("max_low_active_fraction", 0.1)),
            {
                "actual": metrics["low_active_fraction"],
                "maximum": float(global_cfg.get("max_low_active_fraction", 0.1)),
                "active_ratio_threshold": min_active,
            },
        ),
        _check(
            "long_trailing_silence_fraction",
            metrics["long_trailing_silence_fraction"]
            <= float(global_cfg.get("max_long_trailing_silence_fraction", 0.1)),
            {
                "actual": metrics["long_trailing_silence_fraction"],
                "maximum": float(
                    global_cfg.get("max_long_trailing_silence_fraction", 0.1)
                ),
                "seconds_threshold": tail_threshold,
            },
        ),
        _check(
            "long_silence_fraction",
            metrics["long_silence_fraction"]
            <= float(global_cfg.get("max_long_silence_fraction", 0.1)),
            {
                "actual": metrics["long_silence_fraction"],
                "maximum": float(global_cfg.get("max_long_silence_fraction", 0.1)),
                "seconds_threshold": silence_threshold,
            },
        ),
        _check(
            "duplicate_source_id_fraction",
            metrics["duplicate_source_id_fraction"]
            <= float(global_cfg.get("max_duplicate_source_id_fraction", 0.0)),
            {
                "actual": metrics["duplicate_source_id_fraction"],
                "maximum": float(
                    global_cfg.get("max_duplicate_source_id_fraction", 0.0)
                ),
            },
        ),
        _check(
            "sparse_template_fraction",
            metrics["sparse_template_fraction"]
            <= float(sparse_cfg.get("max_fraction", 0.05)),
            {
                "actual": metrics["sparse_template_fraction"],
                "maximum": float(sparse_cfg.get("max_fraction", 0.05)),
                "templates": sorted(sparse_names),
            },
        ),
        _check(
            "repeated_event_has_multiple_spans",
            bool(repeated_rows)
            and metrics["repeated_event_violation_fraction"]
            <= float(repeated_cfg.get("max_violation_fraction", 0.0)),
            {
                "n": len(repeated_rows),
                "actual": metrics["repeated_event_violation_fraction"],
                "maximum": float(repeated_cfg.get("max_violation_fraction", 0.0)),
                "minimum_sfx_spans": min_repeated_spans,
            },
        ),
        _check(
            "overlapping_speakers_overlap",
            bool(overlap_rows)
            and metrics["overlap_violation_fraction"]
            <= float(overlap_cfg.get("max_violation_fraction", 0.1)),
            {
                "n": len(overlap_rows),
                "actual": metrics["overlap_violation_fraction"],
                "maximum": float(overlap_cfg.get("max_violation_fraction", 0.1)),
                "minimum_overlap_ratio": min_overlap,
            },
        ),
    ]

    by_template: dict[str, dict] = {}
    for template in sorted({row["template"] for row in rows}):
        selected = [row for row in rows if row["template"] == template]
        by_template[template] = {
            "n": len(selected),
            "fraction": _fraction(len(selected), total),
            "mean_event_count": _mean(float(row["event_count"]) for row in selected),
            "mean_span_count": _mean(float(row["span_count"]) for row in selected),
            "mean_active_ratio": _mean(float(row["active_ratio"]) for row in selected),
            "mean_trailing_silence_sec": _mean(
                float(row["trailing_silence_sec"]) for row in selected
            ),
            "mean_overlap_ratio": _mean(
                float(row["overlap_ratio"]) for row in selected
            ),
        }

    violation_samples: list[dict] = []
    for row in rows:
        reasons: list[str] = []
        if row["event_count"] <= 1:
            reasons.append("single_event")
        if row["active_ratio"] < min_active:
            reasons.append("low_active_ratio")
        if row["trailing_silence_sec"] > tail_threshold:
            reasons.append("long_trailing_silence")
        if row["max_silence_sec"] > silence_threshold:
            reasons.append("long_silence")
        if row["duplicate_source_ids"]:
            reasons.append("duplicate_source_ids")
        if row["template"] == repeated_name and row["max_sfx_spans"] < min_repeated_spans:
            reasons.append("repeated_event_has_too_few_spans")
        if row["template"] == overlap_name and row["overlap_ratio"] < min_overlap:
            reasons.append("insufficient_speaker_overlap")
        if reasons and len(violation_samples) < 200:
            violation_samples.append({**row, "reasons": reasons})

    passed = all(item["pass"] is True for item in checks)
    return {
        "schema_version": "sceneledger-mixture-quality-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": passed,
        "status": "passed" if passed else "failed",
        "profile": profile_name,
        "quality_config_sha256": config_sha256,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "metrics": metrics,
        "by_template": by_template,
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if item["pass"] is not True],
        "violation_samples": violation_samples,
    }


__all__ = [
    "SPLIT_NAMES",
    "audit_mixture_distribution",
    "build_split_contract",
    "file_sha256",
    "load_quality_profile",
    "require_experiment_data_summary",
    "require_ledger_split",
    "require_split_manifest",
    "write_references",
    "write_split_contract",
]
