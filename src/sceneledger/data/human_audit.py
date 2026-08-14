"""Deterministic, fail-closed human listening audit for rendered mixtures.

The automatic data gate proves replay, source isolation, and coarse distribution
properties.  It cannot prove that a scheduled event is perceptually audible or
that the mixture sounds consistent with its stems.  This module creates a
frozen, template-stratified listening sheet and validates the completed sheet
without silently accepting missing or ambiguous answers.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sceneledger.data.manifests import ManifestEntry, read_manifest
from sceneledger.data.schema import Ledger

AUDIT_SCHEMA_VERSION = "sceneledger-human-audit-v2"

IMMUTABLE_FIELDS = (
    "audit_id",
    "task_id",
    "dataset_id",
    "split",
    "sample_id",
    "template",
    "selection_reason",
    "quality_violation_reasons",
    "duration_sec",
    "mixture_path",
    "stem_paths_json",
    "expected_events_json",
    "speech_review_required",
    "overlap_review_required",
    "stem_review_required",
)

REVIEW_FIELDS = (
    "reviewer",
    "reviewed_at_utc",
    "event_audibility",
    "caption_accuracy",
    "speech_intelligibility",
    "speech_transcript_accuracy",
    "timestamp_alignment",
    "overlap_rendering",
    "long_silence",
    "clipping",
    "stem_mixture_consistency",
    "severity",
    "overall_decision",
    "notes",
)

CSV_FIELDS = IMMUTABLE_FIELDS + REVIEW_FIELDS

PASS_FAIL_UNCERTAIN = {"pass", "fail", "uncertain"}
OPTIONAL_CHECK_VALUES = PASS_FAIL_UNCERTAIN | {"not_required"}
PRESENCE_VALUES = {"absent", "present", "uncertain"}
SEVERITY_VALUES = {"none", "minor", "severe"}


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank(sample_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _event_summary(ledger: Ledger) -> str:
    payload = [
        {
            "id": event.id,
            "type": event.type,
            "track_id": event.track_id,
            "spans": [
                [span.start_sec, span.end_sec]
                for span in event.spans
            ],
            "text": event.text,
        }
        for event in ledger.events
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _has_track_overlap(ledger: Ledger) -> bool:
    spans: list[tuple[float, float, str]] = []
    for track in ledger.tracks:
        spans.extend((span.start_sec, span.end_sec, track.id) for span in track.spans)
    for index, (left_start, left_end, left_id) in enumerate(spans):
        for right_start, right_end, right_id in spans[index + 1 :]:
            if left_id != right_id and min(left_end, right_end) > max(left_start, right_start):
                return True
    return False


def _has_speech(ledger: Ledger) -> bool:
    return any(event.type == "speech" for event in ledger.events)


def _task_payload(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {field: row.get(field, "") for field in IMMUTABLE_FIELDS if field != "audit_id"}
        for row in rows
    ]


def tasks_sha256(rows: list[dict[str, str]]) -> str:
    """Hash immutable task fields, excluding reviewer answers and audit ID."""
    return _canonical_hash(_task_payload(rows))


def build_human_audit(
    manifest_path: str | Path,
    quality_report: dict,
    *,
    dataset_id: str,
    split: str = "test",
    per_template: int = 5,
    max_violation_samples: int = 20,
    seed: str | None = None,
    all_samples: bool = False,
) -> tuple[list[dict[str, str]], dict]:
    """Build deterministic stratified tasks plus quality-violation oversampling."""
    if per_template < 1:
        raise ValueError("per_template must be >= 1")
    if max_violation_samples < 0:
        raise ValueError("max_violation_samples must be >= 0")

    entries = read_manifest(manifest_path)
    by_id: dict[str, ManifestEntry] = {}
    by_template: defaultdict[str, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        sample_id = str(entry.scene["scene_id"])
        if sample_id in by_id:
            raise ValueError(f"duplicate sample ID in audit manifest: {sample_id}")
        by_id[sample_id] = entry
        by_template[str(entry.scene["template"])].append(entry)
    if not by_template:
        raise ValueError("cannot create a human audit from an empty manifest")

    shortfalls = {
        template: len(group)
        for template, group in by_template.items()
        if len(group) < per_template
    }
    if shortfalls and not all_samples:
        raise ValueError(
            "insufficient template coverage for human audit: "
            f"required={per_template}, actual={shortfalls}"
        )

    sampling_seed = seed or dataset_id
    selected_reasons: defaultdict[str, set[str]] = defaultdict(set)
    selected_violation_reasons: defaultdict[str, set[str]] = defaultdict(set)
    for _template, group in sorted(by_template.items()):
        ordered = sorted(
            group,
            key=lambda entry: _rank(str(entry.scene["scene_id"]), sampling_seed),
        )
        chosen = ordered if all_samples else ordered[:per_template]
        for entry in chosen:
            selected_reasons[str(entry.scene["scene_id"])].add(
                "all_samples" if all_samples else "template_stratified"
            )

    violation_rows = quality_report.get("violation_samples", [])
    if not isinstance(violation_rows, list):
        raise ValueError("quality report violation_samples must be a list")
    ordered_violations = sorted(
        violation_rows,
        key=lambda item: _rank(str(item.get("sample_id", "")), sampling_seed),
    )
    for item in ordered_violations[:max_violation_samples]:
        sample_id = str(item.get("sample_id", ""))
        if sample_id not in by_id:
            raise ValueError(f"quality violation sample is absent from manifest: {sample_id}")
        selected_reasons[sample_id].add("quality_violation")
        selected_violation_reasons[sample_id].update(str(x) for x in item.get("reasons", []))

    selected_entries = [by_id[sample_id] for sample_id in selected_reasons]
    selected_entries.sort(
        key=lambda entry: (
            str(entry.scene["template"]),
            _rank(str(entry.scene["scene_id"]), sampling_seed),
        )
    )

    first_by_template: set[str] = set()
    rows: list[dict[str, str]] = []
    for index, entry in enumerate(selected_entries, 1):
        sample_id = str(entry.scene["scene_id"])
        template = str(entry.scene["template"])
        ledger = Ledger.model_validate(entry.target_ledger)
        require_stems = template not in first_by_template
        first_by_template.add(template)
        rows.append(
            {
                "audit_id": "",
                "task_id": f"A{index:04d}",
                "dataset_id": dataset_id,
                "split": split,
                "sample_id": sample_id,
                "template": template,
                "selection_reason": ";".join(sorted(selected_reasons[sample_id])),
                "quality_violation_reasons": ";".join(
                    sorted(selected_violation_reasons[sample_id])
                ),
                "duration_sec": f"{float(entry.scene['duration']):.1f}",
                "mixture_path": entry.mixture_path,
                "stem_paths_json": json.dumps(
                    entry.stem_paths, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "expected_events_json": _event_summary(ledger),
                "speech_review_required": "yes" if _has_speech(ledger) else "no",
                "overlap_review_required": "yes" if _has_track_overlap(ledger) else "no",
                "stem_review_required": "yes" if require_stems else "no",
                **{field: "" for field in REVIEW_FIELDS},
            }
        )

    task_hash = tasks_sha256(rows)
    audit_id = _canonical_hash(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "split": split,
            "per_template": per_template,
            "max_violation_samples": max_violation_samples,
            "seed": sampling_seed,
            "all_samples": all_samples,
            "tasks_sha256": task_hash,
        }
    )
    for row in rows:
        row["audit_id"] = audit_id

    metadata = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_id": audit_id,
        "dataset_id": dataset_id,
        "split": split,
        "manifest_path": str(Path(manifest_path).resolve()),
        "sampling": {
            "per_template": per_template,
            "max_violation_samples": max_violation_samples,
            "seed": sampling_seed,
            "all_samples": all_samples,
        },
        "n_tasks": len(rows),
        "n_quality_violation_tasks": sum(
            "quality_violation" in row["selection_reason"] for row in rows
        ),
        "by_template": dict(sorted(Counter(row["template"] for row in rows).items())),
        "tasks_sha256": task_hash,
        "task_ids": [row["task_id"] for row in rows],
        "sample_ids": [row["sample_id"] for row in rows],
    }
    return rows, metadata


def write_human_audit(
    rows: list[dict[str, str]],
    metadata: dict,
    csv_path: str | Path,
    metadata_path: str | Path,
) -> None:
    output_csv = Path(csv_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_metadata = Path(metadata_path)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_human_audit(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(
                "human audit CSV columns changed; expected "
                f"{list(CSV_FIELDS)}, got {reader.fieldnames}"
            )
        return [dict(row) for row in reader]


def _validate_value(
    row: dict[str, str], field: str, allowed: set[str], errors: list[str]
) -> str:
    value = row.get(field, "").strip().lower()
    if value not in allowed:
        errors.append(f"{row.get('task_id')}: {field}={value!r} not in {sorted(allowed)}")
    return value


def summarize_human_audit(
    review_csv: str | Path,
    metadata_path: str | Path,
    *,
    max_severe: int = 2,
    max_total_failures: int = 2,
    template_failure_threshold: int = 2,
) -> dict:
    """Validate reviewer answers and return a fail-closed human gate summary."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported human audit metadata schema")
    rows = read_human_audit(review_csv)
    if len(rows) != metadata.get("n_tasks"):
        raise ValueError("human audit row count differs from frozen metadata")
    if [row["task_id"] for row in rows] != metadata.get("task_ids"):
        raise ValueError("human audit task IDs or ordering changed")
    if [row["sample_id"] for row in rows] != metadata.get("sample_ids"):
        raise ValueError("human audit sample IDs or ordering changed")
    if any(row["audit_id"] != metadata.get("audit_id") for row in rows):
        raise ValueError("human audit ID changed")
    if tasks_sha256(rows) != metadata.get("tasks_sha256"):
        raise ValueError("immutable human audit task fields changed")

    validation_errors: list[str] = []
    incomplete: list[str] = []
    unresolved: list[str] = []
    inconsistent: list[str] = []
    attention: list[dict[str, object]] = []
    issue_counts: Counter[str] = Counter()
    template_issue_counts: Counter[tuple[str, str]] = Counter()
    severity_counts: Counter[str] = Counter()
    overall_counts: Counter[str] = Counter()

    for row in rows:
        task_id = row["task_id"]
        if not row.get("reviewer", "").strip() or not row.get("reviewed_at_utc", "").strip():
            incomplete.append(task_id)
            continue

        audibility = _validate_value(
            row, "event_audibility", PASS_FAIL_UNCERTAIN, validation_errors
        )
        caption_accuracy = _validate_value(
            row, "caption_accuracy", PASS_FAIL_UNCERTAIN, validation_errors
        )
        speech_intelligibility = _validate_value(
            row, "speech_intelligibility", OPTIONAL_CHECK_VALUES, validation_errors
        )
        speech_transcript = _validate_value(
            row, "speech_transcript_accuracy", OPTIONAL_CHECK_VALUES, validation_errors
        )
        timing = _validate_value(
            row, "timestamp_alignment", PASS_FAIL_UNCERTAIN, validation_errors
        )
        overlap = _validate_value(
            row, "overlap_rendering", OPTIONAL_CHECK_VALUES, validation_errors
        )
        long_silence = _validate_value(
            row, "long_silence", PRESENCE_VALUES, validation_errors
        )
        clipping = _validate_value(row, "clipping", PRESENCE_VALUES, validation_errors)
        stems = _validate_value(
            row, "stem_mixture_consistency", OPTIONAL_CHECK_VALUES, validation_errors
        )
        severity = _validate_value(row, "severity", SEVERITY_VALUES, validation_errors)
        overall = _validate_value(
            row, "overall_decision", PASS_FAIL_UNCERTAIN, validation_errors
        )
        severity_counts[severity] += 1
        overall_counts[overall] += 1

        if row["overlap_review_required"] == "yes" and overlap == "not_required":
            validation_errors.append(f"{task_id}: overlap review is required")
        if row["overlap_review_required"] == "no" and overlap != "not_required":
            validation_errors.append(f"{task_id}: overlap must be not_required")
        if row["stem_review_required"] == "yes" and stems == "not_required":
            validation_errors.append(f"{task_id}: stem review is required")
        if row["stem_review_required"] == "no" and stems != "not_required":
            validation_errors.append(f"{task_id}: stem check must be not_required")
        if row["speech_review_required"] == "yes":
            if speech_intelligibility == "not_required" or speech_transcript == "not_required":
                validation_errors.append(f"{task_id}: speech review fields are required")
        elif speech_intelligibility != "not_required" or speech_transcript != "not_required":
            validation_errors.append(f"{task_id}: speech fields must be not_required")

        checks = {
            "event_audibility": audibility,
            "caption_accuracy": caption_accuracy,
            "speech_intelligibility": speech_intelligibility,
            "speech_transcript_accuracy": speech_transcript,
            "timestamp_alignment": timing,
            "overlap_rendering": overlap,
            "long_silence": long_silence,
            "clipping": clipping,
            "stem_mixture_consistency": stems,
        }
        issues: list[str] = []
        for field, value in checks.items():
            if value == "uncertain":
                unresolved.append(f"{task_id}:{field}")
            is_failure = value == "fail" or (
                field in {"long_silence", "clipping"} and value == "present"
            )
            if is_failure:
                issues.append(field)
                issue_counts[field] += 1
                template_issue_counts[(row["template"], field)] += 1
        if overall == "uncertain":
            unresolved.append(f"{task_id}:overall_decision")
        if issues and overall == "pass":
            inconsistent.append(f"{task_id}: failed checks but overall pass")
        if not issues and overall == "fail":
            inconsistent.append(f"{task_id}: overall fail without a failed check")
        if overall == "fail" and severity == "none":
            inconsistent.append(f"{task_id}: overall fail with severity none")
        if severity == "severe" and overall != "fail":
            inconsistent.append(f"{task_id}: severe issue without overall fail")
        if issues or overall != "pass" or severity != "none":
            attention.append(
                {
                    "task_id": task_id,
                    "sample_id": row["sample_id"],
                    "template": row["template"],
                    "issues": issues,
                    "severity": severity,
                    "overall_decision": overall,
                    "notes": row.get("notes", ""),
                }
            )

    template_failures = [
        {"template": template, "criterion": criterion, "count": count}
        for (template, criterion), count in sorted(template_issue_counts.items())
        if count >= template_failure_threshold
    ]
    n_completed = len(rows) - len(incomplete)
    n_severe = severity_counts["severe"]
    n_total_failures = overall_counts["fail"]
    checks = [
        {"name": "all_tasks_completed", "pass": not incomplete, "detail": incomplete},
        {"name": "answer_values_valid", "pass": not validation_errors, "detail": validation_errors},
        {"name": "no_unresolved_answers", "pass": not unresolved, "detail": unresolved},
        {"name": "answers_consistent", "pass": not inconsistent, "detail": inconsistent},
        {
            "name": "severe_failures_within_limit",
            "pass": n_severe <= max_severe,
            "detail": {"actual": n_severe, "maximum": max_severe},
        },
        {
            "name": "total_failures_within_limit",
            "pass": n_total_failures <= max_total_failures,
            "detail": {"actual": n_total_failures, "maximum": max_total_failures},
        },
        {
            "name": "no_template_systematic_failure",
            "pass": not template_failures,
            "detail": template_failures,
        },
    ]
    passed = all(check["pass"] is True for check in checks)
    criterion_reviewed_counts = {
        field: sum(
            str(row.get(field, "")).strip().lower() != "not_required"
            for row in rows
        )
        for field in (
            "event_audibility",
            "caption_accuracy",
            "speech_intelligibility",
            "speech_transcript_accuracy",
            "timestamp_alignment",
        )
    }
    criterion_pass_rates = {
        field: (
            sum(str(row.get(field, "")).strip().lower() == "pass" for row in rows)
            / reviewed
            if reviewed
            else None
        )
        for field, reviewed in criterion_reviewed_counts.items()
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": passed,
        "status": "passed" if passed else "failed",
        "audit_id": metadata["audit_id"],
        "dataset_id": metadata["dataset_id"],
        "split": metadata["split"],
        "review_csv_path": str(Path(review_csv).resolve()),
        "review_csv_sha256": hashlib.sha256(Path(review_csv).read_bytes()).hexdigest(),
        "n_tasks": len(rows),
        "n_completed": n_completed,
        "severity_counts": dict(severity_counts),
        "overall_counts": dict(overall_counts),
        "criterion_failure_counts": dict(issue_counts),
        "criterion_reviewed_counts": criterion_reviewed_counts,
        "criterion_pass_rates": criterion_pass_rates,
        "template_failures": template_failures,
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if check["pass"] is not True],
        "samples_requiring_attention": attention,
    }


def require_human_audit_summary(
    summary_path: str | Path, *, expected_dataset_id: str | None = None
) -> dict:
    """Require a passed human gate and verify its frozen review CSV."""
    path = Path(summary_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"unsupported human audit summary schema in {path}")
    if payload.get("pass") is not True or payload.get("failed_checks"):
        raise ValueError(
            f"human audit has not passed: {payload.get('failed_checks', [])}"
        )
    if expected_dataset_id is not None and payload.get("dataset_id") != expected_dataset_id:
        raise ValueError("human audit and experiment data gate dataset IDs differ")
    review_path = Path(str(payload.get("review_csv_path", "")))
    if not review_path.is_file():
        raise ValueError(f"human audit review CSV is missing: {review_path}")
    actual_hash = hashlib.sha256(review_path.read_bytes()).hexdigest()
    if payload.get("review_csv_sha256") != actual_hash:
        raise ValueError("human audit review CSV changed after summary creation")
    return payload


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CSV_FIELDS",
    "build_human_audit",
    "read_human_audit",
    "require_human_audit_summary",
    "summarize_human_audit",
    "tasks_sha256",
    "write_human_audit",
]
