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
from sceneledger.data.scene_graph_sampler import Scene
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


def scene_plan_sha256(scenes: Iterable[Scene | dict]) -> str:
    """Hash ordered canonical scene dictionaries without rendered artifacts."""
    payload = [
        scene.to_manifest_dict() if isinstance(scene, Scene) else scene
        for scene in scenes
    ]
    return _identity_sha256(payload)


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
        for leakage_group in source.get("leakage_groups", []):
            if leakage_group not in (None, ""):
                identities.add(f"group:{leakage_group}")
        if group not in (None, "") or source.get("leakage_groups"):
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

    preflight = payload.get("scene_plan_preflight")
    if not isinstance(preflight, dict) or preflight.get("pass") is not True:
        raise ValueError("passed scene-plan preflight is missing from data summary")
    preflight_path = Path(str(preflight.get("path", "")))
    if (
        not preflight_path.is_file()
        or file_sha256(preflight_path) != preflight.get("sha256")
    ):
        raise ValueError("scene-plan preflight changed after data gate")
    preflight_payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight_payload.get("schema_version") != "sceneledger-data-preflight-v1"
        or preflight_payload.get("pass") is not True
        or preflight_payload.get("failed_checks")
    ):
        raise ValueError("scene-plan preflight has not passed")

    quality_config = Path(str(payload.get("quality_config_path", "")))
    if not quality_config.is_file():
        raise ValueError(f"quality config missing: {quality_config}")
    if file_sha256(quality_config) != payload.get("quality_config_sha256"):
        raise ValueError("quality config changed after data gate was created")

    reports = payload.get("quality_reports", {})
    complexity_reports = payload.get("complexity_reports", {})
    recipe_review_reports = payload.get("recipe_review_reports", {})
    references = payload.get("references", {})
    if complexity_reports:
        complexity_config = Path(str(payload.get("complexity_config_path", "")))
        if not complexity_config.is_file():
            raise ValueError(f"complexity config missing: {complexity_config}")
        if file_sha256(complexity_config) != payload.get(
            "complexity_config_sha256"
        ):
            raise ValueError("complexity config changed after data gate was created")
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
        if complexity_reports:
            complexity_item = complexity_reports.get(split)
            if (
                not isinstance(complexity_item, dict)
                or complexity_item.get("pass") is not True
            ):
                raise ValueError(
                    f"complexity report for {split} is missing or failed"
                )
            complexity_path = Path(str(complexity_item.get("path", "")))
            if (
                not complexity_path.is_file()
                or file_sha256(complexity_path) != complexity_item.get("sha256")
            ):
                raise ValueError(
                    f"complexity report for {split} changed after data gate"
                )
            complexity = json.loads(complexity_path.read_text(encoding="utf-8"))
            if complexity.get("pass") is not True:
                raise ValueError(f"complexity report for {split} has not passed")
            if (
                complexity.get("manifest_sha256")
                != contract["splits"][split]["manifest_sha256"]
            ):
                raise ValueError(
                    f"complexity report for {split} audits a different manifest"
                )
        preflight_fold = preflight_payload.get("folds", {}).get(split)
        if not isinstance(preflight_fold, dict) or preflight_fold.get("pass") is not True:
            raise ValueError(f"scene-plan preflight for {split} is missing or failed")
        manifest_entries = read_manifest(contract["splits"][split]["manifest_path"])
        if scene_plan_sha256([entry.scene for entry in manifest_entries]) != preflight_fold.get(
            "scene_plan_sha256"
        ):
            raise ValueError(f"rendered {split} scenes differ from preflight plan")
        if recipe_review_reports:
            review_item = recipe_review_reports.get(split)
            if not isinstance(review_item, dict) or review_item.get("pass") is not True:
                raise ValueError(f"recipe review for {split} is missing or failed")
            review_path = Path(str(review_item.get("path", "")))
            if (
                not review_path.is_file()
                or file_sha256(review_path) != review_item.get("sha256")
            ):
                raise ValueError(f"recipe review for {split} changed after data gate")
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if (
                review.get("pass") is not True
                or review.get("recipe_plan_sha256")
                != review_item.get("recipe_plan_sha256")
            ):
                raise ValueError(f"recipe review for {split} has not passed")
            manifest_recipe_hashes = {
                str(
                    entry.scene.get("recipe_metadata", {}).get(
                        "recipe_plan_sha256"
                    )
                    or ""
                )
                for entry in manifest_entries
            }
            manifest_recipe_hashes.discard("")
            if manifest_recipe_hashes != {review.get("recipe_plan_sha256")}:
                raise ValueError(
                    f"recipe review for {split} audits a different recipe plan"
                )
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
    source_paths = [
        str(source.get("path", "")) for source in entry.scene.get("sources", [])
    ]
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
        "duplicate_source_paths": len(source_paths) != len(set(source_paths)),
    }


def _fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _active_signal(waveform):
    """Return active samples, excluding silence and the PCM-16 floor."""
    import numpy as np

    mono = np.asarray(waveform, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    mono = mono.reshape(-1)
    if not mono.size:
        return mono
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    threshold = max(1.5 / 32768.0, 0.1 * rms)
    return mono[np.abs(mono) >= threshold]


def _rms_dbfs(waveform) -> float:
    import math

    import numpy as np

    active = _active_signal(waveform)
    if not active.size:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(active), dtype=np.float64)))
    return 20.0 * math.log10(max(rms, 1e-12))


def _stem_audibility_report(
    entries: list[ManifestEntry], manifest_path: str | Path, config: dict
) -> tuple[dict, list[dict], list[dict]]:
    """Measure actual persisted stems; scene gain metadata is not evidence."""
    import math

    import numpy as np

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - optional data dependency
        raise RuntimeError("stem audibility checks require soundfile") from exc

    base = Path(manifest_path).resolve().parent
    floors = {
        str(kind): float(value)
        for kind, value in config.get("min_active_rms_dbfs_by_kind", {}).items()
    }
    stem_rows: list[dict] = []
    margin_rows: list[dict] = []
    errors: list[dict] = []
    for entry in entries:
        sample_id = _sample_id(entry)
        source_by_id = {
            str(source.get("source_id")): source
            for source in entry.scene.get("sources", [])
        }
        expected_ids = set(source_by_id)
        observed_ids = {str(source_id) for source_id in entry.stem_paths}
        if expected_ids != observed_ids:
            errors.append(
                {
                    "sample_id": sample_id,
                    "error": "stem/source IDs differ",
                    "missing_stems": sorted(expected_ids - observed_ids),
                    "unexpected_stems": sorted(observed_ids - expected_ids),
                }
            )
        loaded: dict[str, tuple[np.ndarray, int, str]] = {}
        for source_id, path_value in entry.stem_paths.items():
            source = source_by_id.get(str(source_id))
            if source is None:
                errors.append(
                    {"sample_id": sample_id, "source_id": source_id, "error": "unknown source ID"}
                )
                continue
            path = Path(path_value)
            resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
            try:
                resolved.relative_to(base)
            except ValueError:
                errors.append(
                    {"sample_id": sample_id, "source_id": source_id, "error": "stem path escapes manifest root"}
                )
                continue
            try:
                waveform, sample_rate = sf.read(
                    resolved, dtype="float32", always_2d=False
                )
            except Exception as exc:
                errors.append(
                    {"sample_id": sample_id, "source_id": source_id, "error": str(exc)}
                )
                continue
            if int(sample_rate) != int(entry.sample_rate):
                errors.append(
                    {
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "error": f"sample rate {sample_rate} != {entry.sample_rate}",
                    }
                )
                continue
            mono = np.asarray(waveform, dtype=np.float64)
            if mono.ndim == 2:
                mono = mono.mean(axis=1)
            kind = str(source.get("kind", "unknown"))
            rms_dbfs = _rms_dbfs(mono)
            minimum = floors.get(kind)
            stem_rows.append(
                {
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "kind": kind,
                    "active_rms_dbfs": round(rms_dbfs, 4),
                    "minimum_dbfs": minimum,
                    "below_floor": minimum is not None and rms_dbfs < minimum,
                }
            )
            loaded[str(source_id)] = (mono, int(sample_rate), kind)

        speech = [waveform for waveform, _sr, kind in loaded.values() if kind == "speech"]
        competitors = [
            waveform
            for waveform, _sr, kind in loaded.values()
            if kind not in {"speech", "vocal"}
        ]
        if not speech or not competitors:
            continue
        length = max(max(len(item) for item in speech), max(len(item) for item in competitors))

        def _sum(
            items: list[np.ndarray], output_length: int = length
        ) -> np.ndarray:
            output = np.zeros(output_length, dtype=np.float64)
            for item in items:
                output[: len(item)] += item
            return output

        speech_sum, competitor_sum = _sum(speech), _sum(competitors)
        speech_active = _active_signal(speech_sum)
        competitor_active = _active_signal(competitor_sum)
        if not speech_active.size or not competitor_active.size:
            continue
        speech_threshold = max(1.5 / 32768.0, float(np.sqrt(np.mean(speech_active**2))) * 0.1)
        competitor_threshold = max(
            1.5 / 32768.0,
            float(np.sqrt(np.mean(competitor_active**2))) * 0.1,
        )
        overlap = (np.abs(speech_sum) >= speech_threshold) & (
            np.abs(competitor_sum) >= competitor_threshold
        )
        if int(overlap.sum()) < max(1, int(0.1 * entry.sample_rate)):
            continue
        speech_rms = float(np.sqrt(np.mean(speech_sum[overlap] ** 2)))
        competitor_rms = float(np.sqrt(np.mean(competitor_sum[overlap] ** 2)))
        margin = 20.0 * math.log10(max(speech_rms, 1e-12) / max(competitor_rms, 1e-12))
        margin_rows.append(
            {
                "sample_id": sample_id,
                "speech_competitor_margin_db": round(margin, 4),
                "overlap_duration_sec": round(float(overlap.sum()) / entry.sample_rate, 4),
            }
        )

    below_floor = [row for row in stem_rows if row["below_floor"]]
    minimum_margin = float(config.get("min_speech_competitor_margin_db", -120.0))
    low_margin = [
        row
        for row in margin_rows
        if row["speech_competitor_margin_db"] < minimum_margin
    ]
    speech_scene_count = sum(
        any(source.get("kind") == "speech" for source in entry.scene.get("sources", []))
        for entry in entries
    )
    by_kind = {}
    for kind in sorted({str(row["kind"]) for row in stem_rows}):
        values = [float(row["active_rms_dbfs"]) for row in stem_rows if row["kind"] == kind]
        by_kind[kind] = {
            "n": len(values),
            "min_active_rms_dbfs": min(values),
            "mean_active_rms_dbfs": _mean(values),
            "max_active_rms_dbfs": max(values),
        }
    metrics = {
        "n_stems": len(stem_rows),
        "n_below_rms_floor": len(below_floor),
        "below_rms_floor_fraction": _fraction(len(below_floor), len(stem_rows)),
        "n_speech_competitor_overlap_scenes": len(margin_rows),
        "n_speech_scenes": speech_scene_count,
        "speech_overlap_measured_fraction": _fraction(
            len(margin_rows), speech_scene_count
        ),
        "n_low_speech_margin_scenes": len(low_margin),
        "low_speech_margin_fraction": _fraction(len(low_margin), len(margin_rows)),
        "mean_speech_competitor_margin_db": _mean(
            float(row["speech_competitor_margin_db"]) for row in margin_rows
        ),
        "minimum_speech_competitor_margin_db": (
            min(float(row["speech_competitor_margin_db"]) for row in margin_rows)
            if margin_rows
            else None
        ),
        "by_kind": by_kind,
    }
    return metrics, errors, [*below_floor, *low_margin]


def _complexity_metrics_and_checks(rows: list[dict], config: dict) -> tuple[dict, list[dict]]:
    """Summarize and validate simple/medium/complex source-count bands."""
    if not config:
        return {}, []
    total = len(rows)
    simple_max = int(config.get("simple_max_sources", 2))
    complex_min = int(config.get("complex_min_sources", 5))
    if complex_min <= simple_max + 1:
        raise ValueError("complex_min_sources must leave a non-empty medium band")

    simple = [row for row in rows if row["source_count"] <= simple_max]
    medium = [
        row
        for row in rows
        if simple_max < row["source_count"] < complex_min
    ]
    complex_rows = [row for row in rows if row["source_count"] >= complex_min]
    template_counts = Counter(str(row["template"]) for row in rows)
    fractions = {
        "simple_fraction": _fraction(len(simple), total),
        "medium_fraction": _fraction(len(medium), total),
        "complex_fraction": _fraction(len(complex_rows), total),
    }
    mean_source_count = _mean(float(row["source_count"]) for row in rows)

    min_complex_overlap = float(config.get("min_complex_overlap_ratio", 0.0))
    rows_with_overlap = [
        row for row in complex_rows if row.get("overlap_ratio") is not None
    ]
    low_overlap = [
        row
        for row in rows_with_overlap
        if float(row["overlap_ratio"]) < min_complex_overlap
    ]
    low_overlap_fraction = _fraction(len(low_overlap), len(rows_with_overlap))

    metrics = {
        "mean_source_count": mean_source_count,
        **fractions,
        "simple_count": len(simple),
        "medium_count": len(medium),
        "complex_count": len(complex_rows),
        "complex_low_overlap_fraction": low_overlap_fraction,
        "complex_overlap_measured_count": len(rows_with_overlap),
    }
    checks: list[dict] = []
    checks.append(
        _check(
            "mean_source_count",
            mean_source_count >= float(config.get("min_mean_source_count", 0.0)),
            {
                "actual": mean_source_count,
                "minimum": float(config.get("min_mean_source_count", 0.0)),
            },
        )
    )
    for band in ("simple", "medium", "complex"):
        limits = config.get(f"{band}_fraction_range", [0.0, 1.0])
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise ValueError(f"{band}_fraction_range must contain [minimum, maximum]")
        minimum, maximum = (float(limits[0]), float(limits[1]))
        actual = fractions[f"{band}_fraction"]
        checks.append(
            _check(
                f"{band}_source_fraction",
                minimum <= actual <= maximum,
                {"actual": actual, "minimum": minimum, "maximum": maximum},
            )
        )

    required_templates = config.get("required_templates", {})
    if not isinstance(required_templates, dict):
        raise ValueError("complexity.required_templates must be a mapping")
    for template, minimum_fraction in sorted(required_templates.items()):
        actual = _fraction(template_counts[str(template)], total)
        checks.append(
            _check(
                f"required_template:{template}",
                actual >= float(minimum_fraction),
                {"actual": actual, "minimum": float(minimum_fraction)},
            )
        )

    # Scene-plan preflight has no rendered overlap evidence, so this check is
    # intentionally deferred until the manifest audit.
    if rows_with_overlap:
        maximum = float(config.get("max_low_overlap_complex_fraction", 1.0))
        checks.append(
            _check(
                "complex_overlap_fraction",
                low_overlap_fraction <= maximum,
                {
                    "actual": low_overlap_fraction,
                    "maximum": maximum,
                    "minimum_overlap_ratio": min_complex_overlap,
                    "n_complex": len(complex_rows),
                },
            )
        )
    return metrics, checks


def audit_scene_plan_distribution(scenes: Iterable[Scene], profile: dict) -> dict:
    """Fail before waveform rendering when a sampled plan misses complexity targets."""
    scene_list = list(scenes)
    rows = [
        {
            "sample_id": scene.scene_id,
            "template": scene.template,
            "source_count": len(scene.sources),
            "overlap_ratio": None,
        }
        for scene in scene_list
    ]
    metrics, complexity_checks = _complexity_metrics_and_checks(
        rows, profile.get("complexity", {})
    )
    source_cfg = profile.get("source_diversity", {})
    all_sources = [source for scene in scene_list for source in scene.sources]
    by_kind: dict[str, list] = defaultdict(list)
    for source in all_sources:
        by_kind[source.kind].append(source)
    source_metrics: dict[str, dict[str, object]] = {}
    source_checks: list[dict] = []
    truncated_noncontinuous = [
        {
            "sample_id": scene.scene_id,
            "source_id": source.source_id,
            "kind": source.kind,
            "source_end_sec": round(source.onset + float(source.source_duration_sec), 6),
            "scene_duration_sec": scene.duration,
        }
        for scene in scene_list
        for source in scene.sources
        if source.kind not in ("music", "ambience")
        and source.source_duration_sec is not None
        and source.onset + float(source.source_duration_sec) > scene.duration + 1e-6
    ]
    source_checks.append(
        _check(
            "noncontinuous_sources_fit_scene",
            not truncated_noncontinuous,
            truncated_noncontinuous[:50],
        )
    )
    if source_cfg:
        required_provenance = tuple(
            source_cfg.get(
                "required_provenance_fields",
                (
                    "source_group",
                    "source_dataset",
                    "source_license",
                    "annotation_origin",
                    "source_file_sha256",
                    "source_duration_sec",
                ),
            )
        )
        min_unique = source_cfg.get("min_unique_sources_by_kind", {})
        min_groups = source_cfg.get("min_unique_groups_by_kind", {})
        min_datasets = source_cfg.get("min_unique_datasets_by_kind", {})
        min_primary_labels = source_cfg.get("min_unique_primary_labels_by_kind", {})
        dataset_ranges = source_cfg.get("dataset_fraction_ranges_by_kind", {})
        max_reuse = float(source_cfg.get("max_source_reuse_fraction", 1.0))
        provenance_missing: list[dict[str, str]] = []
        for kind, sources in sorted(by_kind.items()):
            path_counts = Counter(source.path for source in sources)
            groups = {source.source_group for source in sources if source.source_group}
            datasets = {
                source.source_dataset for source in sources if source.source_dataset
            }
            dataset_counts = Counter(
                source.source_dataset or "<missing>" for source in sources
            )
            dataset_fractions = {
                dataset: count / len(sources)
                for dataset, count in sorted(dataset_counts.items())
            }
            primary_labels = {
                source.source_labels[0]
                for source in sources
                if source.source_labels
            }
            source_metrics[kind] = {
                "n_slots": len(sources),
                "n_unique_sources": len(path_counts),
                "n_unique_groups": len(groups),
                "n_unique_datasets": len(datasets),
                "datasets": sorted(datasets),
                "dataset_fractions": dataset_fractions,
                "n_unique_primary_labels": len(primary_labels),
                "primary_labels": sorted(primary_labels),
                "max_source_reuse_fraction": max(path_counts.values()) / len(sources),
            }
            source_checks.append(
                _check(
                    f"source_diversity:{kind}:unique_primary_labels",
                    len(primary_labels) >= int(min_primary_labels.get(kind, 0)),
                    {
                        "actual": len(primary_labels),
                        "minimum": int(min_primary_labels.get(kind, 0)),
                        "primary_labels": sorted(primary_labels),
                    },
                )
            )
            source_checks.append(
                _check(
                    f"source_diversity:{kind}:dataset_fractions",
                    all(
                        len(bounds) == 2
                        and float(bounds[0]) <= dataset_fractions.get(dataset, 0.0) <= float(bounds[1])
                        for dataset, bounds in dataset_ranges.get(kind, {}).items()
                    ),
                    {
                        "actual": dataset_fractions,
                        "required_ranges": dataset_ranges.get(kind, {}),
                    },
                )
            )
            source_checks.append(
                _check(
                    f"source_diversity:{kind}:unique_datasets",
                    len(datasets) >= int(min_datasets.get(kind, 0)),
                    {
                        "actual": len(datasets),
                        "minimum": int(min_datasets.get(kind, 0)),
                        "datasets": sorted(datasets),
                    },
                )
            )
            source_checks.append(
                _check(
                    f"source_diversity:{kind}:unique_sources",
                    len(path_counts) >= int(min_unique.get(kind, 0)),
                    {"actual": len(path_counts), "minimum": int(min_unique.get(kind, 0))},
                )
            )
            source_checks.append(
                _check(
                    f"source_diversity:{kind}:unique_groups",
                    len(groups) >= int(min_groups.get(kind, 0)),
                    {"actual": len(groups), "minimum": int(min_groups.get(kind, 0))},
                )
            )
            observed_reuse = max(path_counts.values()) / len(sources)
            source_checks.append(
                _check(
                    f"source_diversity:{kind}:reuse_fraction",
                    observed_reuse <= max_reuse,
                    {"actual": observed_reuse, "maximum": max_reuse},
                )
            )
        for scene in scene_list:
            for source in scene.sources:
                for field in required_provenance:
                    if not getattr(source, field, None):
                        provenance_missing.append(
                            {"sample_id": scene.scene_id, "source_id": source.source_id, "field": field}
                        )
        source_checks.append(
            _check("all_sources_have_provenance", not provenance_missing, provenance_missing[:50])
        )
    checks = [
        _check("scene_plan_nonempty", bool(rows), len(rows)),
        *complexity_checks,
        *source_checks,
    ]
    passed = all(item["pass"] is True for item in checks)
    return {
        "schema_version": "sceneledger-scene-plan-quality-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": passed,
        "status": "passed" if passed else "failed",
        "n_samples": len(rows),
        "scene_plan_sha256": scene_plan_sha256(scene_list),
        "metrics": metrics,
        "source_diversity": source_metrics,
        "template_counts": dict(sorted(Counter(row["template"] for row in rows).items())),
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if item["pass"] is not True],
    }


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
                    "duplicate_source_paths": False,
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
    duplicate_source_path_count = sum(row["duplicate_source_paths"] for row in rows)

    sparse_cfg = profile.get("sparse_templates", {})
    sparse_names = set(sparse_cfg.get("names", ["isolated_sfx"]))
    sparse_count = sum(row["template"] in sparse_names for row in rows)

    repeated_cfg = profile.get("repeated_event", {})
    repeated_required = bool(repeated_cfg.get("required", True))
    repeated_name = str(repeated_cfg.get("template", "repeated_event"))
    repeated_rows = [row for row in rows if row["template"] == repeated_name]
    min_repeated_spans = int(repeated_cfg.get("min_sfx_spans", 2))
    repeated_bad = sum(row["max_sfx_spans"] < min_repeated_spans for row in repeated_rows)

    overlap_cfg = profile.get("overlapping_speakers", {})
    overlap_required = bool(overlap_cfg.get("required", True))
    overlap_name = str(overlap_cfg.get("template", "overlapping_speakers"))
    overlap_rows = [row for row in rows if row["template"] == overlap_name]
    min_overlap = float(overlap_cfg.get("min_overlap_ratio", 0.1))
    overlap_bad = sum(row["overlap_ratio"] < min_overlap for row in overlap_rows)

    complexity_metrics, complexity_checks = _complexity_metrics_and_checks(
        rows, profile.get("complexity", {})
    )
    stem_cfg = profile.get("stem_audibility", {})
    stem_metrics: dict = {}
    stem_errors: list[dict] = []
    stem_violations: list[dict] = []
    stem_checks: list[dict] = []
    if stem_cfg:
        stem_metrics, stem_errors, stem_violations = _stem_audibility_report(
            entries, manifest_path, stem_cfg
        )
        stem_checks = [
            _check("all_stems_readable", not stem_errors, stem_errors[:50]),
            _check(
                "stem_rms_floor_violation_fraction",
                stem_metrics["below_rms_floor_fraction"]
                <= float(stem_cfg.get("max_below_rms_floor_fraction", 0.0)),
                {
                    "actual": stem_metrics["below_rms_floor_fraction"],
                    "maximum": float(
                        stem_cfg.get("max_below_rms_floor_fraction", 0.0)
                    ),
                },
            ),
            _check(
                "speech_overlap_measured_fraction",
                stem_metrics["speech_overlap_measured_fraction"]
                >= float(stem_cfg.get("min_speech_overlap_measured_fraction", 0.0)),
                {
                    "actual": stem_metrics["speech_overlap_measured_fraction"],
                    "minimum": float(
                        stem_cfg.get("min_speech_overlap_measured_fraction", 0.0)
                    ),
                },
            ),
            _check(
                "speech_competitor_margin_violation_fraction",
                stem_metrics["low_speech_margin_fraction"]
                <= float(stem_cfg.get("max_low_speech_margin_fraction", 0.0)),
                {
                    "actual": stem_metrics["low_speech_margin_fraction"],
                    "maximum": float(
                        stem_cfg.get("max_low_speech_margin_fraction", 0.0)
                    ),
                    "minimum_margin_db": float(
                        stem_cfg.get("min_speech_competitor_margin_db", -120.0)
                    ),
                },
            ),
        ]

    metrics = {
        "n_samples": total,
        "single_event_fraction": _fraction(single_count, total),
        "low_active_fraction": _fraction(low_active_count, total),
        "long_trailing_silence_fraction": _fraction(long_tail_count, total),
        "long_silence_fraction": _fraction(long_silence_count, total),
        "duplicate_source_id_fraction": _fraction(duplicate_source_count, total),
        "duplicate_source_path_fraction": _fraction(
            duplicate_source_path_count, total
        ),
        "sparse_template_fraction": _fraction(sparse_count, total),
        "repeated_event_violation_fraction": _fraction(repeated_bad, len(repeated_rows)),
        "overlap_violation_fraction": _fraction(overlap_bad, len(overlap_rows)),
        "mean_event_count": _mean(float(row["event_count"]) for row in rows),
        "mean_active_ratio": _mean(float(row["active_ratio"]) for row in rows),
        "mean_trailing_silence_sec": _mean(
            float(row["trailing_silence_sec"]) for row in rows
        ),
        **complexity_metrics,
        "stem_audibility": stem_metrics,
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
        *complexity_checks,
        *stem_checks,
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
            "duplicate_source_path_fraction",
            metrics["duplicate_source_path_fraction"]
            <= float(global_cfg.get("max_duplicate_source_path_fraction", 0.0)),
            {
                "actual": metrics["duplicate_source_path_fraction"],
                "maximum": float(
                    global_cfg.get("max_duplicate_source_path_fraction", 0.0)
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
            (not repeated_required and not repeated_rows)
            or (
                bool(repeated_rows)
                and metrics["repeated_event_violation_fraction"]
                <= float(repeated_cfg.get("max_violation_fraction", 0.0))
            ),
            {
                "n": len(repeated_rows),
                "actual": metrics["repeated_event_violation_fraction"],
                "maximum": float(repeated_cfg.get("max_violation_fraction", 0.0)),
                "minimum_sfx_spans": min_repeated_spans,
                "required": repeated_required,
            },
        ),
        _check(
            "overlapping_speakers_overlap",
            (not overlap_required and not overlap_rows)
            or (
                bool(overlap_rows)
                and metrics["overlap_violation_fraction"]
                <= float(overlap_cfg.get("max_violation_fraction", 0.1))
            ),
            {
                "n": len(overlap_rows),
                "actual": metrics["overlap_violation_fraction"],
                "maximum": float(overlap_cfg.get("max_violation_fraction", 0.1)),
                "minimum_overlap_ratio": min_overlap,
                "required": overlap_required,
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
    stem_reasons: defaultdict[str, set[str]] = defaultdict(set)
    for item in stem_violations:
        reason = (
            "stem_below_rms_floor"
            if item.get("below_floor") is True
            else "low_speech_competitor_margin"
        )
        stem_reasons[str(item.get("sample_id", ""))].add(reason)
    for row in rows:
        reasons: list[str] = []
        reasons.extend(sorted(stem_reasons.get(str(row["sample_id"]), set())))
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
        if row["duplicate_source_paths"]:
            reasons.append("duplicate_source_paths")
        if row["template"] == repeated_name and row["max_sfx_spans"] < min_repeated_spans:
            reasons.append("repeated_event_has_too_few_spans")
        if row["template"] == overlap_name and row["overlap_ratio"] < min_overlap:
            reasons.append("insufficient_speaker_overlap")
        complexity_cfg = profile.get("complexity", {})
        if complexity_cfg and row["source_count"] >= int(
            complexity_cfg.get("complex_min_sources", 5)
        ) and row["overlap_ratio"] < float(
            complexity_cfg.get("min_complex_overlap_ratio", 0.0)
        ):
            reasons.append("low_complex_overlap")
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
        "stem_audibility_violations": stem_violations[:200],
    }


__all__ = [
    "SPLIT_NAMES",
    "audit_scene_plan_distribution",
    "audit_mixture_distribution",
    "build_split_contract",
    "file_sha256",
    "load_quality_profile",
    "require_experiment_data_summary",
    "require_ledger_split",
    "require_split_manifest",
    "scene_plan_sha256",
    "write_references",
    "write_split_contract",
]
