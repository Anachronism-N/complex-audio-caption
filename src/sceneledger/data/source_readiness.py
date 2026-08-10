"""Audio-level readiness checks for a real B3 source pool.

This gate runs before rendering.  It binds every catalog row to a decodable,
non-silent waveform fingerprint and checks that the selected profile has enough
independent sources and source groups for each required sound kind.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from sceneledger.data.manifests import file_hash
from sceneledger.data.source_catalog import SourceRecord, load_source_catalog

REQUIRED_SOURCE_KINDS = ("speech", "vocal", "music", "sfx", "ambience")
_UNKNOWN_LICENSES = {
    "",
    "unknown",
    "none",
    "n/a",
    "na",
    "tbd",
    "todo",
    "replace_with_dataset_license",
}


def load_readiness_profile(config_path: str | Path, profile: str) -> tuple[dict, str]:
    """Load one versioned source-readiness profile and its config hash."""
    path = Path(config_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), dict):
        raise ValueError(f"source readiness config has no profiles mapping: {path}")
    selected = payload["profiles"].get(profile)
    if not isinstance(selected, dict):
        available = sorted(payload["profiles"])
        raise ValueError(f"unknown source readiness profile {profile!r}; choose {available}")
    merged = {
        "audio": dict(payload.get("audio", {})),
        "kinds": dict(selected.get("kinds", {})),
    }
    return merged, file_hash(path)


def _decoded_audio_probe(path: Path, *, blocksize: int = 65536) -> dict:
    """Decode a waveform once and return content/quality measurements."""
    import soundfile as sf

    decoded_hash = hashlib.sha256()
    total_values = 0
    sum_squares = 0.0
    peak = 0.0
    clipped_values = 0
    finite = True
    with sf.SoundFile(path) as handle:
        sample_rate = int(handle.samplerate)
        channels = int(handle.channels)
        frames = int(handle.frames)
        decoded_hash.update(f"sr={sample_rate};ch={channels};".encode())
        for block in handle.blocks(
            blocksize=blocksize,
            dtype="float32",
            always_2d=True,
        ):
            array = np.ascontiguousarray(block, dtype="<f4")
            finite = finite and bool(np.isfinite(array).all())
            absolute = np.abs(array)
            if absolute.size:
                peak = max(peak, float(absolute.max()))
                clipped_values += int(np.count_nonzero(absolute >= 0.999))
                sum_squares += float(np.square(array, dtype=np.float64).sum())
                total_values += int(array.size)
            decoded_hash.update(array.tobytes())
        duration = frames / sample_rate if sample_rate > 0 else 0.0
        rms = math.sqrt(sum_squares / total_values) if total_values else 0.0
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
        return {
            "byte_sha256": file_hash(path),
            "decoded_sha256": decoded_hash.hexdigest(),
            "duration_sec": round(duration, 6),
            "sample_rate": sample_rate,
            "channels": channels,
            "frames": frames,
            "format": handle.format,
            "subtype": handle.subtype,
            "finite": finite,
            "peak": round(peak, 8),
            "rms_dbfs": round(rms_dbfs, 4),
            "clipped_fraction": round(clipped_values / max(1, total_values), 8),
        }


def _quality_errors(record: SourceRecord, probe: dict, audio_config: dict) -> list[str]:
    kind_config = dict(audio_config.get("per_kind", {}).get(record.kind, {}))
    minimum = float(kind_config.get("min_duration_sec", 0.05))
    maximum = float(kind_config.get("max_duration_sec", 600.0))
    min_rms = float(audio_config.get("min_rms_dbfs", -70.0))
    max_clipped = float(audio_config.get("max_clipped_fraction", 0.1))
    errors: list[str] = []
    duration = float(probe["duration_sec"])
    if int(probe["sample_rate"]) <= 0 or int(probe["frames"]) <= 0:
        errors.append("empty_or_invalid_audio")
    if not bool(probe["finite"]):
        errors.append("non_finite_samples")
    if duration < minimum:
        errors.append(f"duration_below_{minimum:g}s")
    if duration > maximum:
        errors.append(f"duration_above_{maximum:g}s")
    if float(probe["rms_dbfs"]) < min_rms:
        errors.append(f"rms_below_{min_rms:g}dbfs")
    if float(probe["clipped_fraction"]) > max_clipped:
        errors.append(f"clipped_fraction_above_{max_clipped:g}")
    return errors


def _source_pool_id(rows: list[dict]) -> str:
    identity = []
    for row in rows:
        identity.append(
            {
                "decoded_sha256": row.get("decoded_sha256"),
                "kind": row.get("kind"),
                "text": row.get("text"),
                "source_group": row.get("source_group"),
                "identity": row.get("identity"),
                "language": row.get("language"),
                "verbatim": row.get("verbatim"),
                "license": row.get("license"),
                "dataset": row.get("dataset"),
            }
        )
    encoded = json.dumps(
        sorted(identity, key=lambda item: json.dumps(item, sort_keys=True)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_source_pool(
    *,
    catalog_path: str | Path,
    inventory_path: str | Path,
    report_path: str | Path,
    profile_name: str,
    profile_config: dict,
    config_sha256: str,
) -> dict:
    """Probe all files, write a frozen inventory, and return a fail-closed report."""
    catalog = Path(catalog_path).resolve()
    inventory = Path(inventory_path).resolve()
    report_destination = Path(report_path).resolve()
    records = load_source_catalog(catalog, require_files=True)
    audio_config = dict(profile_config.get("audio", {}))
    rows: list[dict] = []
    failures: list[dict] = []

    for record in records:
        errors: list[str] = []
        probe: dict = {}
        try:
            probe = _decoded_audio_probe(Path(record.path))
            errors.extend(_quality_errors(record, probe, audio_config))
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"decode_error:{type(exc).__name__}:{exc}")
        row = {**record.to_dict(), **probe, "ok": not errors, "errors": errors}
        rows.append(row)
        if errors:
            failures.append({"path": record.path, "errors": errors})

    decoded_paths: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        decoded_hash = row.get("decoded_sha256")
        if decoded_hash:
            decoded_paths[str(decoded_hash)].append(str(row["path"]))
    duplicate_groups = [
        paths for paths in decoded_paths.values() if len(paths) > 1
    ]

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    check("all_audio_decoded_and_quality_checked", not failures, failures[:50])
    check("decoded_audio_unique", not duplicate_groups, duplicate_groups[:50])
    unknown_licenses = [
        row["path"]
        for row in rows
        if str(row.get("license") or "").strip().lower() in _UNKNOWN_LICENSES
    ]
    check("all_licenses_known", not unknown_licenses, unknown_licenses[:50])
    vocal_errors = [
        row["path"]
        for row in rows
        if row["kind"] == "vocal" and row.get("verbatim") is not True
    ]
    check("all_vocal_lyrics_verbatim", not vocal_errors, vocal_errors[:50])

    kind_summary: dict[str, dict] = {}
    requirements = dict(profile_config.get("kinds", {}))
    for kind in REQUIRED_SOURCE_KINDS:
        kind_rows = [row for row in rows if row["kind"] == kind]
        requirement = dict(requirements.get(kind, {}))
        n_sources = len(kind_rows)
        n_groups = len({str(row["source_group"]) for row in kind_rows})
        total_duration = round(
            sum(float(row.get("duration_sec", 0.0)) for row in kind_rows), 3
        )
        min_sources = int(requirement.get("min_sources", 1))
        min_groups = int(requirement.get("min_source_groups", 1))
        min_duration = float(requirement.get("min_total_duration_sec", 0.0))
        kind_summary[kind] = {
            "n_sources": n_sources,
            "n_source_groups": n_groups,
            "total_duration_sec": total_duration,
            "requirements": {
                "min_sources": min_sources,
                "min_source_groups": min_groups,
                "min_total_duration_sec": min_duration,
            },
        }
        check(
            f"{kind}_source_quota",
            n_sources >= min_sources,
            {"required": min_sources, "actual": n_sources},
        )
        check(
            f"{kind}_group_quota",
            n_groups >= min_groups,
            {"required": min_groups, "actual": n_groups},
        )
        check(
            f"{kind}_duration_quota",
            total_duration >= min_duration,
            {"required": min_duration, "actual": total_duration},
        )

    rows.sort(key=lambda row: (str(row["kind"]), str(row["source_group"]), str(row["path"])))
    inventory.parent.mkdir(parents=True, exist_ok=True)
    with inventory.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    passed = bool(checks) and all(item["pass"] is True for item in checks)
    summary = {
        "schema_version": "b3-source-readiness-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": passed,
        "status": "passed" if passed else "failed",
        "profile": profile_name,
        "profile_config_sha256": config_sha256,
        "source_catalog_path": str(catalog),
        "source_catalog_sha256": file_hash(catalog),
        "inventory_path": str(inventory),
        "inventory_sha256": file_hash(inventory),
        "source_pool_id": _source_pool_id(rows),
        "n_sources": len(rows),
        "n_audio_ok": sum(bool(row["ok"]) for row in rows),
        "n_unique_decoded_audio": len(decoded_paths),
        "duplicate_decoded_audio_groups": duplicate_groups,
        "kinds": kind_summary,
        "datasets": dict(sorted(Counter(row.get("dataset") or "unknown" for row in rows).items())),
        "licenses": dict(sorted(Counter(row.get("license") or "unknown" for row in rows).items())),
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if item["pass"] is not True],
    }
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def require_source_readiness_summary(
    path: str | Path, *, expected_profile: str | None = None
) -> dict:
    """Load a passed source report or raise before rendering starts."""
    report = Path(path).resolve()
    if not report.is_file():
        raise FileNotFoundError(f"source readiness report missing: {report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {report}")
    if payload.get("pass") is not True or payload.get("failed_checks"):
        raise ValueError(
            f"source pool readiness has not passed: {payload.get('failed_checks', [])}"
        )
    if not payload.get("source_pool_id"):
        raise ValueError("source readiness report has no source_pool_id")
    if expected_profile and payload.get("profile") != expected_profile:
        raise ValueError(
            f"source readiness profile {payload.get('profile')} != expected {expected_profile}"
        )
    return payload


__all__ = [
    "REQUIRED_SOURCE_KINDS",
    "audit_source_pool",
    "load_readiness_profile",
    "require_source_readiness_summary",
]
