"""Validated real single-source catalog and leakage-safe grouped splitting.

The renderer can only provide exact mixture timestamps when every dry source is
traceable.  This module therefore treats a source catalog as a first-class
experiment artifact rather than a loose ``{kind: [path]}`` dictionary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

CATALOG_SCHEMA_VERSION = "sceneledger.source_catalog.v1"
CatalogKind = Literal["speech", "vocal", "music", "sfx", "ambience"]
AnnotationOrigin = Literal["human", "dataset", "asr", "audio_model", "llm_rewrite"]
FOLDS = ("train", "val", "test")
SOURCE_AUDIT_IMMUTABLE_FIELDS = (
    "source_id",
    "split",
    "kind",
    "audio_path",
    "caption",
    "identity",
    "text_is_verbatim",
)


class SourceRecord(BaseModel):
    """One auditable dry recording or stem used by the mixture renderer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[CATALOG_SCHEMA_VERSION] = CATALOG_SCHEMA_VERSION
    source_id: str = Field(..., min_length=1)
    kind: CatalogKind
    audio_path: str = Field(..., min_length=1)
    source_group: str = Field(..., min_length=1)
    leakage_groups: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    caption: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    license: str = Field(..., min_length=1)
    annotation_origin: AnnotationOrigin
    text_is_verbatim: bool = False
    identity: str | None = None
    language: str | None = None
    attribution: str | None = None
    original_url: str | None = None
    duration_sec: float | None = Field(default=None, gt=0.0)
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    file_sha256: str | None = None
    content_sha256: str | None = None
    rms_dbfs: float | None = None
    active_rms_dbfs: float | None = None
    clipped_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    split: Literal["train", "val", "test"] | None = None

    @field_validator(
        "source_id",
        "audio_path",
        "source_group",
        "caption",
        "dataset",
        "license",
        mode="before",
    )
    @classmethod
    def _strip_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("leakage_groups", "labels")
    @classmethod
    def _normalize_leakage_groups(cls, value: list[str]) -> list[str]:
        normalized = sorted({item.strip() for item in value if item.strip()})
        return normalized


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_audit_tasks_sha256(rows: list[dict[str, str]]) -> str:
    payload = [
        {field: str(row.get(field, "")) for field in SOURCE_AUDIT_IMMUTABLE_FIELDS}
        for row in rows
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_source_catalog(path: str | Path) -> list[SourceRecord]:
    """Read strict JSONL and include the line number in validation failures."""
    records: list[SourceRecord] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(SourceRecord.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid source catalog {path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"source catalog is empty: {path}")
    return records


def write_source_catalog(path: str | Path, records: list[SourceRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda item: item.source_id):
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False, sort_keys=True) + "\n")


def _sample_audio_windows(path: Path, window_sec: float) -> tuple[np.ndarray, int, int, float]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "source catalog audio checks require soundfile; install with `pip install -e .[data]`"
        ) from exc

    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("audio has no decodable samples")
    window = max(1, min(info.frames, int(round(window_sec * info.samplerate))))
    starts = sorted({0, max(0, (info.frames - window) // 2), max(0, info.frames - window)})
    pieces: list[np.ndarray] = []
    with sf.SoundFile(path) as handle:
        for start in starts:
            handle.seek(start)
            piece = handle.read(window, dtype="float32", always_2d=True)
            if piece.size:
                pieces.append(piece.mean(axis=1))
    if not pieces:
        raise ValueError("audio decoder returned no samples")
    samples = np.concatenate(pieces).astype(np.float32, copy=False)
    return samples, int(info.samplerate), int(info.channels), float(info.duration)


def _content_fingerprint(samples: np.ndarray, sample_rate: int) -> str:
    """Hash deterministic waveform windows after mono/8 kHz normalization."""
    from scipy.signal import resample_poly

    target_rate = 8000
    divisor = math.gcd(sample_rate, target_rate)
    if sample_rate != target_rate:
        samples = resample_poly(samples, target_rate // divisor, sample_rate // divisor)
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def probe_source_record(
    record: SourceRecord,
    *,
    audio_root: str | Path,
    fingerprint_window_sec: float = 10.0,
) -> SourceRecord:
    """Resolve, decode and fingerprint a record while keeping a portable path."""
    root = Path(audio_root).expanduser().resolve()
    raw_path = Path(record.audio_path).expanduser()
    resolved = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
    try:
        portable_path = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"audio path escapes --audio-root: {record.audio_path}") from exc
    if not resolved.is_file():
        raise ValueError(f"audio file does not exist: {resolved}")

    samples, sample_rate, channels, duration = _sample_audio_windows(
        resolved, fingerprint_window_sec
    )
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + 1e-12))
    peak = np.abs(samples)
    active_threshold = max(1.5 / 32768.0, rms * 0.1)
    active = samples[peak >= active_threshold]
    active_rms = (
        float(np.sqrt(np.mean(np.square(active, dtype=np.float64)) + 1e-12))
        if active.size
        else 0.0
    )
    return record.model_copy(
        update={
            "audio_path": portable_path,
            "duration_sec": round(duration, 6),
            "sample_rate": sample_rate,
            "channels": channels,
            "file_sha256": file_sha256(resolved),
            "content_sha256": _content_fingerprint(samples, sample_rate),
            "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 4),
            "active_rms_dbfs": round(
                20.0 * math.log10(max(active_rms, 1e-12)), 4
            ),
            "clipped_fraction": round(float(np.mean(peak >= 0.999)), 8),
        }
    )


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _split_groups(
    records: list[SourceRecord],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[SourceRecord]]:
    """Greedy stratified split; a source_group is assigned exactly once."""
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must contain three positive values")
    total_ratio = sum(ratios)
    ratio_by_fold = {fold: ratios[i] / total_ratio for i, fold in enumerate(FOLDS)}
    groups = _connected_source_groups(records)

    fixed_assignments: dict[str, str] = {}
    for group, group_records in groups.items():
        hints = {record.split for record in group_records if record.split is not None}
        if len(hints) > 1:
            raise ValueError(
                f"source_group {group!r} has conflicting fixed split hints: {sorted(hints)}"
            )
        if hints:
            fixed_assignments[group] = next(iter(hints))

    total_by_kind = Counter(record.kind for record in records)
    target_rows = {fold: len(records) * ratio_by_fold[fold] for fold in FOLDS}
    target_kind = {
        fold: {kind: count * ratio_by_fold[fold] for kind, count in total_by_kind.items()}
        for fold in FOLDS
    }
    counts = {fold: Counter() for fold in FOLDS}
    row_counts = Counter()
    assignments: dict[str, str] = dict(fixed_assignments)
    for group, fold in fixed_assignments.items():
        row_counts[fold] += len(groups[group])
        counts[fold].update(record.kind for record in groups[group])

    def tie_hash(group: str, fold: str) -> str:
        return hashlib.sha256(f"{seed}:{group}:{fold}".encode()).hexdigest()

    ordered_groups = sorted(
        (group for group in groups if group not in fixed_assignments),
        key=lambda group: (-len(groups[group]), tie_hash(group, "order")),
    )
    for group in ordered_groups:
        group_kind = Counter(record.kind for record in groups[group])
        candidates: list[tuple[float, str, str]] = []
        for candidate in FOLDS:
            score = 0.0
            for fold in FOLDS:
                rows = row_counts[fold] + (len(groups[group]) if fold == candidate else 0)
                score += ((rows - target_rows[fold]) / max(1.0, target_rows[fold])) ** 2
                for kind in total_by_kind:
                    value = counts[fold][kind] + (group_kind[kind] if fold == candidate else 0)
                    target = target_kind[fold][kind]
                    score += ((value - target) / max(1.0, target)) ** 2
            candidates.append((score, tie_hash(group, candidate), candidate))
        fold = min(candidates)[2]
        assignments[group] = fold
        row_counts[fold] += len(groups[group])
        counts[fold].update(group_kind)

    output = {fold: [] for fold in FOLDS}
    for group, group_records in groups.items():
        fold = assignments[group]
        for record in group_records:
            output[fold].append(record.model_copy(update={"split": fold}))
    return output


def _record_group_tokens(record: SourceRecord) -> set[str]:
    tokens = {record.source_group, *record.leakage_groups}
    if record.identity:
        tokens.add(f"identity:{record.identity}")
    return tokens


def _connected_source_groups(records: list[SourceRecord]) -> dict[str, list[SourceRecord]]:
    """Collapse records sharing any leakage identity into connected components."""
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    for record in records:
        tokens = sorted(_record_group_tokens(record))
        for token in tokens:
            find(token)
        for token in tokens[1:]:
            union(tokens[0], token)

    component_tokens: dict[str, list[str]] = defaultdict(list)
    for token in parent:
        component_tokens[find(token)].append(token)
    canonical = {
        root: min(tokens) for root, tokens in component_tokens.items()
    }
    output: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        output[canonical[find(record.source_group)]].append(record)
    return output


def _check(name: str, passed: bool, detail: object) -> dict[str, object]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def _catalog_checks(
    records: list[SourceRecord],
    splits: dict[str, list[SourceRecord]],
    *,
    allowed_licenses: set[str],
    min_duration_sec: float,
    max_clipped_fraction: float,
    min_rms_dbfs: float,
    min_records_per_kind_per_split: int,
    min_groups_per_kind_per_split: int,
    min_caption_unique_fraction: float,
    required_kinds: set[str],
) -> list[dict[str, object]]:
    ids = [record.source_id for record in records]
    paths = [record.audio_path for record in records]
    file_hashes = [record.file_sha256 or "" for record in records]
    content_hashes = [record.content_sha256 or "" for record in records]
    licenses = sorted({record.license for record in records})
    caption_diversity = {
        kind: len({record.caption for record in records if record.kind == kind})
        / sum(record.kind == kind for record in records)
        for kind in sorted({record.kind for record in records})
    }
    checks = [
        _check("source_ids_unique", not _duplicates(ids), _duplicates(ids)[:20]),
        _check("audio_paths_unique", not _duplicates(paths), _duplicates(paths)[:20]),
        _check(
            "file_hashes_unique",
            not _duplicates(file_hashes),
            _duplicates(file_hashes)[:20],
        ),
        _check(
            "content_hashes_unique",
            not _duplicates(content_hashes),
            _duplicates(content_hashes)[:20],
        ),
        _check(
            "licenses_allowlisted",
            bool(allowed_licenses) and set(licenses) <= allowed_licenses,
            {"observed": licenses, "allowed": sorted(allowed_licenses)},
        ),
        _check(
            "all_source_kinds_present",
            required_kinds <= {record.kind for record in records},
            sorted(required_kinds - {record.kind for record in records}),
        ),
        _check(
            "caption_diversity",
            all(value >= min_caption_unique_fraction for value in caption_diversity.values()),
            {
                "observed": {kind: round(value, 6) for kind, value in caption_diversity.items()},
                "minimum": min_caption_unique_fraction,
            },
        ),
        _check(
            "minimum_duration",
            all((record.duration_sec or 0.0) >= min_duration_sec for record in records),
            [record.source_id for record in records if (record.duration_sec or 0.0) < min_duration_sec][
                :20
            ],
        ),
        _check(
            "not_silent",
            all((record.rms_dbfs or -999.0) >= min_rms_dbfs for record in records),
            [record.source_id for record in records if (record.rms_dbfs or -999.0) < min_rms_dbfs][
                :20
            ],
        ),
        _check(
            "not_clipped",
            all((record.clipped_fraction or 0.0) <= max_clipped_fraction for record in records),
            [
                record.source_id
                for record in records
                if (record.clipped_fraction or 0.0) > max_clipped_fraction
            ][:20],
        ),
    ]
    group_sets = {
        fold: {token for record in fold_records for token in _record_group_tokens(record)}
        for fold, fold_records in splits.items()
    }
    hash_sets = {
        fold: {record.content_sha256 for record in fold_records}
        for fold, fold_records in splits.items()
    }
    for fold in FOLDS:
        checks.append(_check(f"{fold}_nonempty", bool(splits[fold]), len(splits[fold])))
        kind_counts = Counter(record.kind for record in splits[fold])
        kind_group_counts = {
            kind: len(
                _connected_source_groups(
                    [record for record in splits[fold] if record.kind == kind]
                )
            )
            for kind in required_kinds
        }
        checks.append(
            _check(
                f"{fold}_minimum_per_kind",
                all(kind_counts[kind] >= min_records_per_kind_per_split for kind in required_kinds),
                {
                    "observed": dict(sorted(kind_counts.items())),
                    "minimum": min_records_per_kind_per_split,
                },
            )
        )
        checks.append(
            _check(
                f"{fold}_minimum_groups_per_kind",
                all(
                    kind_group_counts[kind] >= min_groups_per_kind_per_split
                    for kind in required_kinds
                ),
                {
                    "observed": dict(sorted(kind_group_counts.items())),
                    "minimum": min_groups_per_kind_per_split,
                },
            )
        )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        group_overlap = sorted(group_sets[left] & group_sets[right])
        hash_overlap = sorted(value for value in hash_sets[left] & hash_sets[right] if value)
        checks.append(_check(f"{left}_{right}_groups_disjoint", not group_overlap, group_overlap[:20]))
        checks.append(
            _check(f"{left}_{right}_content_disjoint", not hash_overlap, hash_overlap[:20])
        )
    return checks


def _write_audit_sheet(
    path: Path,
    splits: dict[str, list[SourceRecord]],
    *,
    per_kind: int,
    seed: int,
) -> None:
    candidates: dict[str, dict[str, list[SourceRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for fold in FOLDS:
        for record in splits[fold]:
            candidates[record.kind][fold].append(record)
    chosen: list[SourceRecord] = []
    for _kind, by_fold in sorted(candidates.items()):
        queues = {
            fold: sorted(
                by_fold[fold],
                key=lambda record: hashlib.sha256(
                    f"{seed}:audit:{record.source_id}".encode()
                ).hexdigest(),
            )
            for fold in FOLDS
        }
        # Round-robin makes a 10-row kind audit cover train/val/test as
        # 4/3/3, and a 30-row audit as 10/10/10.  A global hash sample could
        # otherwise miss the exact test fold used by a pilot.
        selected_for_kind = 0
        while selected_for_kind < per_kind and any(queues.values()):
            for fold in FOLDS:
                if selected_for_kind >= per_kind:
                    break
                if queues[fold]:
                    chosen.append(queues[fold].pop(0))
                    selected_for_kind += 1
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_id",
                "split",
                "kind",
                "audio_path",
                "caption",
                "identity",
                "text_is_verbatim",
                "audible_y_n",
                "caption_correct_y_n",
                "kind_correct_y_n",
                "notes",
            ]
        )
        for record in chosen:
            writer.writerow(
                [
                    record.source_id,
                    record.split,
                    record.kind,
                    record.audio_path,
                    record.caption,
                    record.identity or "",
                    str(record.text_is_verbatim).lower(),
                    "",
                    "",
                    "",
                    "",
                ]
            )


def prepare_source_catalog(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    audio_root: str | Path,
    allowed_licenses: set[str],
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 20260813,
    fingerprint_window_sec: float = 10.0,
    min_duration_sec: float = 0.2,
    min_rms_dbfs: float = -55.0,
    max_clipped_fraction: float = 0.01,
    audit_per_kind: int = 10,
    min_records_per_kind_per_split: int = 4,
    min_groups_per_kind_per_split: int = 4,
    min_caption_unique_fraction: float = 0.5,
    required_kinds: set[str] | None = None,
) -> dict[str, object]:
    """Normalize raw metadata, split by source_group, and write a gate report."""
    raw_records = read_source_catalog(input_path)
    expected_kinds = required_kinds or {"speech", "vocal", "music", "sfx", "ambience"}
    invalid_kinds = sorted(expected_kinds - {"speech", "vocal", "music", "sfx", "ambience"})
    if invalid_kinds:
        raise ValueError(f"unsupported required source kinds: {invalid_kinds}")
    records: list[SourceRecord] = []
    probe_errors: list[dict[str, str]] = []
    for record in raw_records:
        try:
            records.append(
                probe_source_record(
                    record,
                    audio_root=audio_root,
                    fingerprint_window_sec=fingerprint_window_sec,
                )
            )
        except Exception as exc:
            probe_errors.append({"source_id": record.source_id, "error": str(exc)})

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if probe_errors:
        report: dict[str, object] = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "pass": False,
            "input_path": str(input_path),
            "audio_root": str(Path(audio_root).resolve()),
            "n_input": len(raw_records),
            "n_probed": len(records),
            "probe_errors": probe_errors,
            "checks": [_check("all_audio_decodable", False, probe_errors[:20])],
        }
        (output / "source_catalog_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    splits = _split_groups(records, split_ratios, seed)
    checks = [_check("all_audio_decodable", True, len(records))]
    checks.extend(
        _catalog_checks(
            records,
            splits,
            allowed_licenses=allowed_licenses,
            min_duration_sec=min_duration_sec,
            max_clipped_fraction=max_clipped_fraction,
            min_rms_dbfs=min_rms_dbfs,
            min_records_per_kind_per_split=min_records_per_kind_per_split,
            min_groups_per_kind_per_split=min_groups_per_kind_per_split,
            min_caption_unique_fraction=min_caption_unique_fraction,
            required_kinds=expected_kinds,
        )
    )
    report = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "pass": all(bool(check["pass"]) for check in checks),
        "input_path": str(input_path),
        "audio_root": str(Path(audio_root).resolve()),
        "seed": seed,
        "split_ratios": {fold: split_ratios[index] for index, fold in enumerate(FOLDS)},
        "n_records": len(records),
        "counts_by_kind": dict(sorted(Counter(record.kind for record in records).items())),
        "required_kinds": sorted(expected_kinds),
        "counts_by_split": {
            fold: {
                "records": len(splits[fold]),
                "groups": len(_connected_source_groups(splits[fold])),
                "by_kind": dict(sorted(Counter(record.kind for record in splits[fold]).items())),
            }
            for fold in FOLDS
        },
        "caption_unique_fraction_by_kind": {
            kind: round(
                len({record.caption for record in records if record.kind == kind})
                / sum(record.kind == kind for record in records),
                6,
            )
            for kind in sorted({record.kind for record in records})
        },
        "checks": checks,
    }
    (output / "source_catalog_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not report["pass"]:
        return report

    write_source_catalog(output / "all.jsonl", [record for fold in FOLDS for record in splits[fold]])
    for fold in FOLDS:
        write_source_catalog(output / f"{fold}.jsonl", splits[fold])
    _write_audit_sheet(output / "source_audit.csv", splits, per_kind=audit_per_kind, seed=seed)
    with (output / "source_audit.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        frozen_audit_rows = list(csv.DictReader(handle))
    report["source_audit_n_tasks"] = len(frozen_audit_rows)
    report["source_audit_tasks_sha256"] = _source_audit_tasks_sha256(
        frozen_audit_rows
    )
    report["artifacts"] = {
        name: {
            "path": name,
            "sha256": file_sha256(output / name),
        }
        for name in ("all.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "source_audit.csv")
    }
    (output / "source_catalog_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def validate_source_audit(
    preparation_report_path: str | Path,
    audit_csv_path: str | Path,
    output_path: str | Path,
    *,
    min_per_kind: int = 10,
    min_pass_rate: float = 0.9,
    required_splits: set[str] | None = None,
    min_per_kind_per_required_split: int = 3,
) -> dict[str, object]:
    """Validate completed human source review and bind it to frozen catalogs."""
    prep_path = Path(preparation_report_path)
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = [
        _check("preparation_gate_passed", prep.get("pass") is True, prep.get("pass"))
    ]
    artifacts = prep.get("artifacts") or {}
    catalog_hash_checks: dict[str, dict[str, object]] = {}
    for name in ("all.jsonl", "train.jsonl", "val.jsonl", "test.jsonl"):
        metadata = artifacts.get(name) or {}
        artifact_path = Path(metadata.get("path", prep_path.parent / name))
        if not artifact_path.is_absolute():
            artifact_path = (prep_path.parent / artifact_path).resolve()
        expected = metadata.get("sha256")
        observed = file_sha256(artifact_path) if artifact_path.is_file() else None
        catalog_hash_checks[name] = {
            "path": str(artifact_path),
            "sha256": observed,
            "expected_sha256": expected,
        }
        checks.append(
            _check(
                f"{name}_matches_preparation",
                bool(expected) and observed == expected,
                catalog_hash_checks[name],
            )
        )

    all_catalog_path = Path(catalog_hash_checks["all.jsonl"]["path"])
    known = {record.source_id: record for record in read_source_catalog(all_catalog_path)}
    with Path(audit_csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {
        *SOURCE_AUDIT_IMMUTABLE_FIELDS,
        "audible_y_n",
        "caption_correct_y_n",
        "kind_correct_y_n",
    }
    observed_columns = set(rows[0]) if rows else set()
    checks.append(
        _check(
            "audit_columns_present",
            required_columns <= observed_columns,
            sorted(required_columns - observed_columns),
        )
    )
    ids = [str(row.get("source_id", "")).strip() for row in rows]
    checks.append(_check("audit_nonempty", bool(rows), len(rows)))
    checks.append(
        _check(
            "audit_tasks_frozen",
            len(rows) == prep.get("source_audit_n_tasks")
            and _source_audit_tasks_sha256(rows)
            == prep.get("source_audit_tasks_sha256"),
            {
                "actual_n_tasks": len(rows),
                "expected_n_tasks": prep.get("source_audit_n_tasks"),
                "actual_tasks_sha256": _source_audit_tasks_sha256(rows),
                "expected_tasks_sha256": prep.get("source_audit_tasks_sha256"),
            },
        )
    )
    checks.append(_check("audit_source_ids_unique", not _duplicates(ids), _duplicates(ids)[:20]))
    unknown = sorted(set(ids) - set(known))
    checks.append(_check("audit_sources_in_catalog", not unknown, unknown[:20]))

    fields = ("audible_y_n", "caption_correct_y_n", "kind_correct_y_n")
    normalized: dict[str, list[bool]] = {field: [] for field in fields}
    normalized_by_split: dict[str, dict[str, list[bool]]] = {
        split: {field: [] for field in fields} for split in FOLDS
    }
    invalid_answers: list[dict[str, str]] = []
    rejected_source_ids: set[str] = set()
    for row in rows:
        for field in fields:
            answer = str(row.get(field, "")).strip().lower()
            if answer not in {"y", "n"}:
                invalid_answers.append(
                    {"source_id": str(row.get("source_id", "")), "field": field, "value": answer}
                )
            else:
                passed = answer == "y"
                if not passed:
                    rejected_source_ids.add(
                        str(row.get("source_id", "")).strip()
                    )
                normalized[field].append(passed)
                source_id = str(row.get("source_id", "")).strip()
                if source_id in known and known[source_id].split in FOLDS:
                    normalized_by_split[str(known[source_id].split)][field].append(passed)
    checks.append(_check("audit_complete", not invalid_answers, invalid_answers[:20]))

    counts_by_kind = Counter(
        known[source_id].kind for source_id in ids if source_id in known
    )
    counts_by_split_kind = {
        split: Counter(
            known[source_id].kind
            for source_id in ids
            if source_id in known and known[source_id].split == split
        )
        for split in FOLDS
    }
    required_kinds = set(prep.get("required_kinds") or {"speech", "vocal", "music", "sfx", "ambience"})
    checks.append(
        _check(
            "audit_minimum_per_kind",
            all(counts_by_kind[kind] >= min_per_kind for kind in required_kinds),
            {"observed": dict(sorted(counts_by_kind.items())), "minimum": min_per_kind},
        )
    )
    pass_rates = {
        field: (sum(values) / len(rows) if len(values) == len(rows) and rows else 0.0)
        for field, values in normalized.items()
    }
    pass_rates_by_split = {
        split: {
            field: (
                sum(values) / len(values)
                if len(values) == sum(counts_by_split_kind[split].values()) and values
                else 0.0
            )
            for field, values in normalized_by_split[split].items()
        }
        for split in FOLDS
    }
    checks.append(
        _check(
            "audit_pass_rates",
            all(rate >= min_pass_rate for rate in pass_rates.values()),
            {"observed": pass_rates, "minimum": min_pass_rate},
        )
    )
    requested_splits = set(required_splits or set())
    invalid_splits = requested_splits - set(FOLDS)
    if invalid_splits:
        raise ValueError(f"unsupported required source-audit splits: {sorted(invalid_splits)}")
    for split in sorted(requested_splits):
        checks.append(
            _check(
                f"audit_minimum_per_kind:{split}",
                all(
                    counts_by_split_kind[split][kind]
                    >= min_per_kind_per_required_split
                    for kind in required_kinds
                ),
                {
                    "observed": dict(sorted(counts_by_split_kind[split].items())),
                    "minimum": min_per_kind_per_required_split,
                },
            )
        )
        checks.append(
            _check(
                f"audit_pass_rates:{split}",
                all(rate >= min_pass_rate for rate in pass_rates_by_split[split].values()),
                {"observed": pass_rates_by_split[split], "minimum": min_pass_rate},
            )
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": "sceneledger.source_audit.v1",
        "pass": all(bool(check["pass"]) for check in checks),
        "preparation_report_path": os.path.relpath(
            prep_path.resolve(), destination.parent.resolve()
        ).replace("\\", "/"),
        "preparation_report_sha256": file_sha256(prep_path),
        "audit_csv_path": str(audit_csv_path),
        "audit_csv_sha256": file_sha256(audit_csv_path),
        "catalog_artifacts": catalog_hash_checks,
        "n_reviewed": len(rows),
        "counts_by_kind": dict(sorted(counts_by_kind.items())),
        "counts_by_split_kind": {
            split: dict(sorted(counts.items()))
            for split, counts in counts_by_split_kind.items()
        },
        "pass_rates": pass_rates,
        "pass_rates_by_split": pass_rates_by_split,
        # Reviewed failures are quarantined by CatalogSourcePool even when the
        # aggregate sampling audit still passes its confidence threshold.
        "rejected_source_ids": sorted(rejected_source_ids),
        "required_splits": sorted(requested_splits),
        "checks": checks,
    }
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "FOLDS",
    "SourceRecord",
    "file_sha256",
    "prepare_source_catalog",
    "probe_source_record",
    "read_source_catalog",
    "validate_source_audit",
    "write_source_catalog",
]
