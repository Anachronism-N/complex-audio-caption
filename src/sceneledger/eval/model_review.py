"""Contract-bound, blinded human comparison of two captioning systems."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sceneledger.data.experiment_data import (
    file_sha256,
    require_experiment_data_summary,
    require_ledger_split,
    require_split_manifest,
)
from sceneledger.data.manifests import read_manifest
from sceneledger.data.schema import Ledger

REVIEW_SCHEMA_VERSION = "sceneledger-model-review-v1"

IMMUTABLE_FIELDS = (
    "review_id",
    "task_id",
    "dataset_id",
    "sample_id",
    "template",
    "duration_sec",
    "audio_path",
    "candidate_a_json",
    "candidate_b_json",
)

REVIEW_FIELDS = (
    "reviewer",
    "reviewed_at_utc",
    "a_semantic_support_1_5",
    "a_completeness_1_5",
    "a_temporal_alignment_1_5",
    "a_source_attribution_1_5_or_na",
    "a_hallucination_count",
    "a_omission_count",
    "b_semantic_support_1_5",
    "b_completeness_1_5",
    "b_temporal_alignment_1_5",
    "b_source_attribution_1_5_or_na",
    "b_hallucination_count",
    "b_omission_count",
    "preference_a_b_tie",
    "notes",
)

CSV_FIELDS = IMMUTABLE_FIELDS + REVIEW_FIELDS


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_ledgers(path: str | Path) -> dict[str, Ledger]:
    result: dict[str, Ledger] = {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        ledger = Ledger.model_validate(json.loads(line))
        if ledger.sample_id in result:
            raise ValueError(f"duplicate sample ID in {path}:{line_no}: {ledger.sample_id}")
        result[ledger.sample_id] = ledger
    return result


def _event_json(ledger: Ledger) -> str:
    events = [
        {
            "type": event.type,
            "spans": [
                [span.start_sec, span.end_sec]
                for span in event.spans
            ],
            "text": event.text,
            "track_id": event.track_id,
        }
        for event in ledger.events
    ]
    return json.dumps(events, ensure_ascii=False, separators=(",", ":"))


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _select_stratified(entries: list[Any], count: int, seed: str) -> list[Any]:
    if count < 1:
        raise ValueError("review sample count must be >= 1")
    if count > len(entries):
        raise ValueError(f"requested {count} review samples from only {len(entries)} test rows")
    groups: defaultdict[str, list[Any]] = defaultdict(list)
    for entry in entries:
        groups[str(entry.scene.get("template", "unknown"))].append(entry)
    for template in groups:
        groups[template].sort(
            key=lambda entry: _rank(seed, str(entry.scene["scene_id"]))
        )
    selected: list[Any] = []
    offset = 0
    templates = sorted(groups)
    while len(selected) < count:
        added = False
        for template in templates:
            if offset < len(groups[template]):
                selected.append(groups[template][offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    return selected


def _load_inference_report(
    path: str | Path,
    *,
    prediction_path: str | Path,
    dataset_id: str,
    expected_ids: set[str],
) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema_version") != "sceneledger-inference-report-v1":
        raise ValueError(f"unsupported inference report: {path}")
    if report.get("dataset_id") != dataset_id or report.get("expected_split") != "test":
        raise ValueError(f"inference report is not bound to frozen test: {path}")
    if report.get("prediction_sha256") != file_sha256(prediction_path):
        raise ValueError(f"prediction hash differs from inference report: {path}")
    report_ids = [str(row.get("sample_id", "")) for row in report.get("samples", [])]
    if len(report_ids) != len(set(report_ids)) or set(report_ids) != expected_ids:
        raise ValueError(f"inference report IDs do not equal frozen test: {path}")
    if report.get("n_samples") != len(expected_ids):
        raise ValueError(f"inference report sample count is inconsistent: {path}")
    return report


def _tasks_sha256(rows: list[dict[str, str]]) -> str:
    payload = [
        {field: row.get(field, "") for field in IMMUTABLE_FIELDS if field != "review_id"}
        for row in rows
    ]
    return _canonical_hash(payload)


def prepare_model_review(
    *,
    manifest_path: str | Path,
    audio_base: str | Path,
    zero_predictions_path: str | Path,
    zero_inference_report_path: str | Path,
    tuned_predictions_path: str | Path,
    tuned_inference_report_path: str | Path,
    validity_audit_path: str | Path,
    split_contract_path: str | Path,
    data_gate_summary_path: str | Path,
    sample_count: int = 60,
    seed: str = "sceneledger-model-review-v1",
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Prepare randomized A/B tasks after proving both arms use frozen test."""
    gate = require_experiment_data_summary(data_gate_summary_path, split_contract_path)
    contract = require_split_manifest(split_contract_path, "test", manifest_path)
    dataset_id = str(gate["dataset_id"])
    if contract.get("dataset_id") != dataset_id:
        raise ValueError("data gate and split contract dataset IDs differ")
    require_ledger_split(
        split_contract_path, "test", zero_predictions_path, role="zero-shot prediction"
    )
    require_ledger_split(
        split_contract_path, "test", tuned_predictions_path, role="tuned prediction"
    )

    entries = read_manifest(manifest_path)
    expected_ids = {str(entry.scene["scene_id"]) for entry in entries}
    _load_inference_report(
        zero_inference_report_path,
        prediction_path=zero_predictions_path,
        dataset_id=dataset_id,
        expected_ids=expected_ids,
    )
    _load_inference_report(
        tuned_inference_report_path,
        prediction_path=tuned_predictions_path,
        dataset_id=dataset_id,
        expected_ids=expected_ids,
    )

    validity = json.loads(Path(validity_audit_path).read_text(encoding="utf-8"))
    if (
        validity.get("schema_version") != "sceneledger-evaluation-validity-v1"
        or validity.get("pass") is not True
        or validity.get("status") != "certified_generalization"
        or validity.get("dataset_id") != dataset_id
    ):
        raise ValueError("tuned result does not have passed generalization certification")
    tuned_report_hash = file_sha256(tuned_inference_report_path)
    if validity.get("artifacts", {}).get("inference_report", {}).get(
        "sha256"
    ) != tuned_report_hash:
        raise ValueError("validity audit is bound to a different tuned inference report")

    zero_ledgers = _read_ledgers(zero_predictions_path)
    tuned_ledgers = _read_ledgers(tuned_predictions_path)
    selected = _select_stratified(entries, sample_count, seed)
    audio_root = Path(audio_base).expanduser().resolve()
    rows: list[dict[str, str]] = []
    assignments: list[dict[str, str]] = []
    for index, entry in enumerate(selected, 1):
        sample_id = str(entry.scene["scene_id"])
        audio_path = (audio_root / entry.mixture_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"review audio is missing: {audio_path}")
        zero_text = _event_json(zero_ledgers[sample_id])
        tuned_text = _event_json(tuned_ledgers[sample_id])
        tuned_is_a = int(_rank(f"{seed}:swap", sample_id), 16) % 2 == 0
        arm_a = "b3_tuned" if tuned_is_a else "zero_shot"
        arm_b = "zero_shot" if tuned_is_a else "b3_tuned"
        candidate_a = tuned_text if tuned_is_a else zero_text
        candidate_b = zero_text if tuned_is_a else tuned_text
        task_id = f"MR{index:04d}"
        rows.append(
            {
                "review_id": "",
                "task_id": task_id,
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "template": str(entry.scene.get("template", "unknown")),
                "duration_sec": f"{float(entry.scene['duration']):.1f}",
                "audio_path": str(audio_path),
                "candidate_a_json": candidate_a,
                "candidate_b_json": candidate_b,
                **{field: "" for field in REVIEW_FIELDS},
            }
        )
        assignments.append(
            {"task_id": task_id, "sample_id": sample_id, "arm_a": arm_a, "arm_b": arm_b}
        )

    tasks_hash = _tasks_sha256(rows)
    review_id = _canonical_hash(
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "seed": seed,
            "sample_count": sample_count,
            "tasks_sha256": tasks_hash,
            "zero_prediction_sha256": file_sha256(zero_predictions_path),
            "tuned_prediction_sha256": file_sha256(tuned_predictions_path),
        }
    )
    for row in rows:
        row["review_id"] = review_id

    key = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_id": review_id,
        "dataset_id": dataset_id,
        "assignments": assignments,
    }
    metadata = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_id": review_id,
        "dataset_id": dataset_id,
        "split": "test",
        "seed": seed,
        "n_tasks": len(rows),
        "tasks_sha256": tasks_hash,
        "task_ids": [row["task_id"] for row in rows],
        "sample_ids": [row["sample_id"] for row in rows],
        "by_template": dict(sorted(Counter(row["template"] for row in rows).items())),
        "artifacts": {
            "manifest": file_sha256(manifest_path),
            "split_contract": file_sha256(split_contract_path),
            "data_gate_summary": file_sha256(data_gate_summary_path),
            "validity_audit": file_sha256(validity_audit_path),
            "zero_predictions": file_sha256(zero_predictions_path),
            "zero_inference_report": file_sha256(zero_inference_report_path),
            "tuned_predictions": file_sha256(tuned_predictions_path),
            "tuned_inference_report": tuned_report_hash,
        },
        "instructions": {
            "rating_scale": "1=unsupported/wrong, 3=partly correct, 5=fully correct",
            "source_attribution": "use na only when the clip has no attribution question",
            "preference": "a, b, or tie after listening; do not inspect the key file",
        },
    }
    return rows, metadata, key


def write_model_review(
    rows: list[dict[str, str]],
    metadata: dict[str, Any],
    key: dict[str, Any],
    *,
    csv_path: str | Path,
    metadata_path: str | Path,
    key_path: str | Path,
) -> None:
    output_csv = Path(csv_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_key = Path(key_path)
    output_key.parent.mkdir(parents=True, exist_ok=True)
    output_key.write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = dict(metadata)
    metadata["key_sha256"] = file_sha256(output_key)
    output_metadata = Path(metadata_path)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_review_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("model review CSV columns changed")
        return [dict(row) for row in reader]


def _rating(row: dict[str, str], field: str) -> float:
    raw = row.get(field, "").strip().lower()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{row['task_id']}: {field} must be an integer 1..5") from exc
    if value not in range(1, 6):
        raise ValueError(f"{row['task_id']}: {field} must be in 1..5")
    return float(value)


def _attribution(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "").strip().lower()
    if raw == "na":
        return None
    return _rating(row, field)


def _count(row: dict[str, str], field: str) -> int:
    raw = row.get(field, "").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{row['task_id']}: {field} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{row['task_id']}: {field} must be non-negative")
    return value


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _two_sided_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return round(min(1.0, 2 * probability), 8)


def summarize_model_review(
    *,
    review_csv_path: str | Path | list[str | Path],
    metadata_path: str | Path,
    key_path: str | Path,
) -> dict[str, Any]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported model review metadata schema")
    if key.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported model review key schema")
    if file_sha256(key_path) != metadata.get("key_sha256"):
        raise ValueError("model review key changed after task preparation")
    if key.get("review_id") != metadata.get("review_id"):
        raise ValueError("model review key and metadata IDs differ")
    review_paths = (
        [Path(path) for path in review_csv_path]
        if isinstance(review_csv_path, list)
        else [Path(review_csv_path)]
    )
    if not review_paths:
        raise ValueError("at least one completed model review CSV is required")
    review_hashes = [file_sha256(path) for path in review_paths]
    if len(review_hashes) != len(set(review_hashes)):
        raise ValueError("duplicate completed review files were supplied")
    review_batches = [_read_review_csv(path) for path in review_paths]
    for rows in review_batches:
        if len(rows) != metadata.get("n_tasks"):
            raise ValueError("model review row count changed")
        if [row["task_id"] for row in rows] != metadata.get("task_ids"):
            raise ValueError("model review task IDs or order changed")
        if [row["sample_id"] for row in rows] != metadata.get("sample_ids"):
            raise ValueError("model review sample IDs or order changed")
        if any(row["review_id"] != metadata.get("review_id") for row in rows):
            raise ValueError("model review ID changed")
        if _tasks_sha256(rows) != metadata.get("tasks_sha256"):
            raise ValueError("immutable model review fields changed")
    rows = [row for batch in review_batches for row in batch]

    assignments = {row["task_id"]: row for row in key.get("assignments", [])}
    if set(assignments) != set(metadata.get("task_ids", [])):
        raise ValueError("model review key assignments do not match tasks")
    arm_values: dict[str, defaultdict[str, list[float]]] = {
        "zero_shot": defaultdict(list),
        "b3_tuned": defaultdict(list),
    }
    arm_counts: dict[str, Counter[str]] = {
        "zero_shot": Counter(),
        "b3_tuned": Counter(),
    }
    preference = Counter()
    preference_by_task: defaultdict[str, list[str]] = defaultdict(list)
    reviewers: set[str] = set()
    paired_deltas: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not row.get("reviewer", "").strip() or not row.get(
            "reviewed_at_utc", ""
        ).strip():
            raise ValueError(f"{row['task_id']}: reviewer and reviewed_at_utc are required")
        reviewers.add(row["reviewer"].strip())
        candidate_scores: dict[str, dict[str, float | None]] = {}
        for side in ("a", "b"):
            candidate_scores[side] = {
                "semantic_support": _rating(row, f"{side}_semantic_support_1_5"),
                "completeness": _rating(row, f"{side}_completeness_1_5"),
                "temporal_alignment": _rating(row, f"{side}_temporal_alignment_1_5"),
                "source_attribution": _attribution(
                    row, f"{side}_source_attribution_1_5_or_na"
                ),
            }
            candidate_scores[side]["hallucination_count"] = float(
                _count(row, f"{side}_hallucination_count")
            )
            candidate_scores[side]["omission_count"] = float(
                _count(row, f"{side}_omission_count")
            )
        assignment = assignments[row["task_id"]]
        score_by_arm = {
            assignment["arm_a"]: candidate_scores["a"],
            assignment["arm_b"]: candidate_scores["b"],
        }
        for arm, scores in score_by_arm.items():
            for metric, value in scores.items():
                if value is not None:
                    if metric.endswith("_count"):
                        arm_counts[arm][metric] += int(value)
                    else:
                        arm_values[arm][metric].append(float(value))
        for metric in (
            "semantic_support",
            "completeness",
            "temporal_alignment",
            "source_attribution",
        ):
            zero_value = score_by_arm["zero_shot"][metric]
            tuned_value = score_by_arm["b3_tuned"][metric]
            if zero_value is not None and tuned_value is not None:
                paired_deltas[metric].append(float(tuned_value) - float(zero_value))
        raw_preference = row.get("preference_a_b_tie", "").strip().lower()
        if raw_preference not in {"a", "b", "tie"}:
            raise ValueError(f"{row['task_id']}: preference must be a, b, or tie")
        actual_preference = (
            "tie" if raw_preference == "tie" else assignment[f"arm_{raw_preference}"]
        )
        preference[actual_preference] += 1
        preference_by_task[row["task_id"]].append(actual_preference)

    if len(review_batches) > 1:
        batch_reviewers = [
            {row["reviewer"].strip() for row in batch if row["reviewer"].strip()}
            for batch in review_batches
        ]
        if any(len(names) != 1 for names in batch_reviewers):
            raise ValueError("each completed review CSV must contain exactly one reviewer")
        names = [next(iter(values)) for values in batch_reviewers]
        if len(names) != len(set(names)):
            raise ValueError("completed review CSVs must come from distinct reviewers")

    arm_summary: dict[str, Any] = {}
    for arm in ("zero_shot", "b3_tuned"):
        arm_summary[arm] = {
            "n_samples": len(rows),
            **{f"mean_{metric}": _mean(values) for metric, values in arm_values[arm].items()},
            "total_hallucination": arm_counts[arm]["hallucination_count"],
            "total_omission": arm_counts[arm]["omission_count"],
        }
    deltas = {f"mean_{metric}": _mean(values) for metric, values in paired_deltas.items()}
    consensus = Counter()
    for labels in preference_by_task.values():
        counts = Counter(labels)
        if counts["b3_tuned"] > counts["zero_shot"]:
            consensus["b3_tuned"] += 1
        elif counts["zero_shot"] > counts["b3_tuned"]:
            consensus["zero_shot"] += 1
        else:
            consensus["tie"] += 1
    tuned_wins = consensus["b3_tuned"]
    zero_wins = consensus["zero_shot"]
    no_regression = all(
        deltas.get(f"mean_{metric}") is None
        or deltas[f"mean_{metric}"] >= 0
        for metric in ("semantic_support", "completeness", "temporal_alignment")
    )
    hallucination_safe = (
        arm_summary["b3_tuned"]["total_hallucination"]
        <= arm_summary["zero_shot"]["total_hallucination"]
    )
    omission_safe = (
        arm_summary["b3_tuned"]["total_omission"]
        <= arm_summary["zero_shot"]["total_omission"]
    )
    go_for_scale = (
        no_regression
        and hallucination_safe
        and omission_safe
        and tuned_wins > zero_wins
    )
    agreement_pairs = 0
    agreement_matches = 0
    for labels in preference_by_task.values():
        for left_index, left in enumerate(labels):
            for right in labels[left_index + 1 :]:
                agreement_pairs += 1
                agreement_matches += int(left == right)
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "pass": True,
        "status": "complete",
        "review_id": metadata["review_id"],
        "dataset_id": metadata["dataset_id"],
        "n_tasks": int(metadata["n_tasks"]),
        "n_completed_reviews": len(review_batches),
        "n_reviewers": len(reviewers),
        "n_judgments": len(rows),
        "arms": arm_summary,
        "paired_delta_tuned_minus_zero": deltas,
        "preference_judgments": dict(
            (arm, preference[arm]) for arm in ("b3_tuned", "zero_shot", "tie")
        ),
        "preference_sample_consensus": {
            "b3_tuned": tuned_wins,
            "zero_shot": zero_wins,
            "tie": consensus["tie"],
            "two_sided_sign_test_p": _two_sided_sign_p(tuned_wins, zero_wins),
            "pairwise_reviewer_agreement": (
                round(agreement_matches / agreement_pairs, 6)
                if agreement_pairs
                else None
            ),
            "n_agreement_pairs": agreement_pairs,
        },
        "go_for_scale": go_for_scale,
        "decision_checks": {
            "semantic_completeness_timing_no_regression": no_regression,
            "hallucination_not_increased": hallucination_safe,
            "omission_not_increased": omission_safe,
            "tuned_preferred_more_often": tuned_wins > zero_wins,
        },
        "artifacts": {
            "review_csvs": [
                {"path": str(path.resolve()), "sha256": digest}
                for path, digest in zip(review_paths, review_hashes, strict=True)
            ],
            "metadata_sha256": file_sha256(metadata_path),
            "key_sha256": file_sha256(key_path),
        },
    }


__all__ = [
    "CSV_FIELDS",
    "prepare_model_review",
    "summarize_model_review",
    "write_model_review",
]
