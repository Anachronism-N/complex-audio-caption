"""Manifest-only audit for temporal, source and modality complexity.

This gate intentionally does not inspect model predictions.  It answers
whether a frozen data artifact actually contains the multi-source evidence a
complex-audio model is supposed to learn from, and whether every placed source
retains enough provenance for leakage-safe splitting.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from sceneledger.data.manifests import ManifestEntry, read_manifest


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _track_spans(entry: ManifestEntry) -> list[tuple[float, float, str]]:
    duration = float(entry.scene["duration"])
    spans: list[tuple[float, float, str]] = []
    for track in entry.target_ledger.get("tracks", []):
        kind = str(track.get("kind") or "unknown")
        for span in track.get("spans", []):
            start = max(0.0, min(duration, float(span["start_sec"])))
            end = max(0.0, min(duration, float(span["end_sec"])))
            if end > start:
                spans.append((start, end, kind))
    return spans


def _temporal_stats(
    spans: list[tuple[float, float, str]], duration: float
) -> dict[str, float | int]:
    boundaries = sorted({0.0, duration, *(x for span in spans for x in span[:2])})
    active_duration = 0.0
    overlap_duration = 0.0
    max_simultaneous = 0
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        simultaneous = sum(left <= midpoint < right for left, right, _ in spans)
        max_simultaneous = max(max_simultaneous, simultaneous)
        if simultaneous:
            active_duration += end - start
        if simultaneous >= 2:
            overlap_duration += end - start
    interior_transitions = sum(0.0 < boundary < duration for boundary in boundaries)
    return {
        "active_ratio": round(active_duration / duration, 6) if duration else 0.0,
        "overlap_ratio": round(overlap_duration / duration, 6) if duration else 0.0,
        "max_simultaneous_tracks": max_simultaneous,
        "temporal_transitions": interior_transitions,
    }


def _source_provenance(source: dict[str, Any]) -> bool:
    required = (
        "source_group",
        "source_dataset",
        "source_file_sha256",
        "source_labels",
    )
    if any(not source.get(field) for field in required):
        return False
    path = str(source.get("path") or "")
    return bool(path) and not path.startswith("real:")


def _scene_stats(
    entry: ManifestEntry,
    *,
    complex_definition: dict[str, float | int],
    simple_definition: dict[str, float | int],
) -> dict[str, Any]:
    duration = float(entry.scene["duration"])
    sources = list(entry.scene.get("sources", []))
    events = list(entry.target_ledger.get("events", []))
    tracks = list(entry.target_ledger.get("tracks", []))
    temporal = _temporal_stats(_track_spans(entry), duration)
    kinds = Counter(str(source.get("kind") or "unknown") for source in sources)
    n_sources = len(sources)
    n_events = len(events)
    overlap = float(temporal["overlap_ratio"])
    max_simultaneous = int(temporal["max_simultaneous_tracks"])
    is_complex = (
        n_sources >= int(complex_definition.get("min_sources", 4))
        and n_events >= int(complex_definition.get("min_events", 4))
        and overlap >= float(complex_definition.get("min_overlap_ratio", 0.15))
        and max_simultaneous
        >= int(complex_definition.get("min_simultaneous_tracks", 2))
    )
    is_simple = (
        n_sources <= int(simple_definition.get("max_sources", 2))
        and n_events <= int(simple_definition.get("max_events", 2))
        and overlap <= float(simple_definition.get("max_overlap_ratio", 0.05))
    )
    sequential_only = n_events >= 2 and max_simultaneous <= 1
    block_like = (
        n_events <= int(simple_definition.get("max_events", 2))
        and int(temporal["temporal_transitions"])
        <= int(simple_definition.get("max_temporal_transitions", 4))
    )
    return {
        "scene_id": str(entry.scene["scene_id"]),
        "template": str(entry.scene.get("template") or "unknown"),
        "duration_sec": duration,
        "n_sources": n_sources,
        "n_tracks": len(tracks),
        "n_events": n_events,
        "n_kinds": len(kinds),
        "n_voice_tracks": kinds["speech"] + kinds["vocal"],
        "n_sfx_sources": kinds["sfx"],
        "has_speech_music_sfx": bool(kinds["speech"] and kinds["music"] and kinds["sfx"]),
        "has_music_vocal": bool(kinds["music"] and kinds["vocal"]),
        "provenance_complete": bool(sources) and all(
            _source_provenance(source) for source in sources
        ),
        "is_complex": is_complex,
        "is_simple": is_simple,
        "is_sequential_only": sequential_only,
        "is_block_like": block_like,
        **temporal,
    }


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
    }


def audit_manifest_complexity(
    manifest_path: str | Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Audit one frozen manifest against a pre-registered complexity profile."""
    entries = read_manifest(manifest_path)
    if not entries:
        raise ValueError("complexity audit requires a non-empty manifest")
    complex_definition = dict(profile.get("complex_definition") or {})
    simple_definition = dict(profile.get("simple_definition") or {})
    gates = dict(profile.get("gates") or {})
    rows = [
        _scene_stats(
            entry,
            complex_definition=complex_definition,
            simple_definition=simple_definition,
        )
        for entry in entries
    ]
    n = len(rows)
    summary = {
        "n_scenes": n,
        "mean_sources": round(mean(row["n_sources"] for row in rows), 6),
        "mean_events": round(mean(row["n_events"] for row in rows), 6),
        "mean_overlap_ratio": round(mean(row["overlap_ratio"] for row in rows), 6),
        "mean_active_ratio": round(mean(row["active_ratio"] for row in rows), 6),
        "complex_scene_fraction": _fraction(sum(row["is_complex"] for row in rows), n),
        "simple_scene_fraction": _fraction(sum(row["is_simple"] for row in rows), n),
        "sequential_only_fraction": _fraction(
            sum(row["is_sequential_only"] for row in rows), n
        ),
        "block_like_fraction": _fraction(sum(row["is_block_like"] for row in rows), n),
        "multi_voice_scene_fraction": _fraction(
            sum(row["n_voice_tracks"] >= 2 for row in rows), n
        ),
        "speech_music_sfx_fraction": _fraction(
            sum(row["has_speech_music_sfx"] for row in rows), n
        ),
        "music_vocal_fraction": _fraction(sum(row["has_music_vocal"] for row in rows), n),
        "provenance_complete_fraction": _fraction(
            sum(row["provenance_complete"] for row in rows), n
        ),
        "source_count_histogram": dict(
            sorted(Counter(row["n_sources"] for row in rows).items())
        ),
        "event_count_histogram": dict(
            sorted(Counter(row["n_events"] for row in rows).items())
        ),
        "simultaneous_track_histogram": dict(
            sorted(Counter(row["max_simultaneous_tracks"] for row in rows).items())
        ),
        "template_counts": dict(
            sorted(Counter(row["template"] for row in rows).items())
        ),
    }
    checks: list[dict[str, Any]] = []
    minimum_metrics = {
        "min_scenes": "n_scenes",
        "min_mean_sources": "mean_sources",
        "min_mean_events": "mean_events",
        "min_mean_overlap_ratio": "mean_overlap_ratio",
        "min_complex_scene_fraction": "complex_scene_fraction",
        "min_multi_voice_scene_fraction": "multi_voice_scene_fraction",
        "min_speech_music_sfx_fraction": "speech_music_sfx_fraction",
        "min_music_vocal_fraction": "music_vocal_fraction",
        "min_provenance_complete_fraction": "provenance_complete_fraction",
    }
    maximum_metrics = {
        "max_simple_scene_fraction": "simple_scene_fraction",
        "max_sequential_only_fraction": "sequential_only_fraction",
        "max_block_like_fraction": "block_like_fraction",
    }
    for gate, metric in minimum_metrics.items():
        if gate in gates:
            checks.append(
                _check(
                    gate,
                    float(summary[metric]) >= float(gates[gate]),
                    summary[metric],
                    {">=": gates[gate]},
                )
            )
    for gate, metric in maximum_metrics.items():
        if gate in gates:
            checks.append(
                _check(
                    gate,
                    float(summary[metric]) <= float(gates[gate]),
                    summary[metric],
                    {"<=": gates[gate]},
                )
            )
    groups_by_kind: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        for source in entry.scene.get("sources", []):
            group = source.get("source_group")
            if group:
                groups_by_kind[str(source.get("kind") or "unknown")].add(str(group))
    observed_groups = {
        kind: len(values) for kind, values in sorted(groups_by_kind.items())
    }
    for kind, minimum in sorted(
        (gates.get("min_unique_source_groups_by_kind") or {}).items()
    ):
        observed = observed_groups.get(str(kind), 0)
        checks.append(
            _check(
                f"min_unique_source_groups:{kind}",
                observed >= int(minimum),
                observed,
                {">=": int(minimum)},
            )
        )
    flagged = [
        row
        for row in rows
        if row["is_simple"]
        or row["is_sequential_only"]
        or row["is_block_like"]
        or not row["provenance_complete"]
    ]
    return {
        "schema_version": "sceneledger.complexity_audit.v1",
        "profile": profile.get("description", "unnamed"),
        "manifest_path": str(manifest_path),
        "pass": all(check["pass"] for check in checks),
        "complex_definition": complex_definition,
        "simple_definition": simple_definition,
        "summary": summary,
        "unique_source_groups_by_kind": observed_groups,
        "checks": checks,
        "flagged_scenes": flagged[:25],
    }


def load_complexity_profile(
    config_path: str | Path, profile_name: str
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(
            f"unknown complexity profile {profile_name!r}; "
            f"choose one of {sorted(profiles or {})}"
        )
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"complexity profile must be a mapping: {profile_name}")
    return profile


def write_complexity_report(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


__all__ = [
    "audit_manifest_complexity",
    "load_complexity_profile",
    "write_complexity_report",
]
