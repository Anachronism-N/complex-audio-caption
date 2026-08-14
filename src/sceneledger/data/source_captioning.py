"""Frozen zero-shot source-caption experiment and human evidence gate."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from sceneledger.data.source_catalog import file_sha256, read_source_catalog
from sceneledger.eval.parser import parse_model_output
from sceneledger.models.moss_adapter import AudioCaptioner
from sceneledger.models.target_formatter import atomic_to_ledger

PLAN_SCHEMA = "sceneledger.source_caption_plan.v1"
RESULT_SCHEMA = "sceneledger.source_caption_results.v1"
AUDIT_SCHEMA = "sceneledger.source_caption_audit.v1"

SEMANTIC_PROMPT = (
    "Describe every audible sound in this clip in precise, natural language. "
    "Mention sound identity, acoustic qualities, order, overlap and repetition only when audible. "
    "Do not guess visual context or invent an exact time that cannot be heard."
)
STRUCTURED_PROMPT = (
    "Describe every audible event using only acoustic evidence. Return only typed events. "
    "Use <speech>, <lys>, <music>, or <sfx> tags and atomic 0.1-second boundary tokens, for example "
    "<sfx><|t_003|>a door closes<|t_012|></sfx>. Events may overlap and repeat. "
    "Do not omit audible events or invent unsupported sources, words, or times."
)
DEFAULT_PROMPTS = {"semantic": SEMANTIC_PROMPT, "structured": STRUCTURED_PROMPT}

AUDIT_FIELDS = [
    "source_id",
    "label",
    "kind",
    "prompt_mode",
    "audio_path",
    "raw_text",
    "error",
    "parsed_event_count",
    "format_parseable_auto",
    "label_correct_y_n",
    "all_audible_events_covered_y_n",
    "hallucination_free_y_n",
    "temporal_claims_present_y_n",
    "temporal_claims_supported_y_n_or_na",
    "structured_format_usable_y_n_or_na",
    "corrected_caption",
    "notes",
]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def build_source_caption_plan(
    catalog_path: str | Path,
    audio_root: str | Path,
    output_path: str | Path,
    *,
    per_label: int = 1,
    seed: int = 20260813,
    prompt_modes: tuple[str, ...] = ("semantic", "structured"),
) -> dict[str, Any]:
    """Select the same deterministic examples for each prompt condition."""
    if per_label <= 0:
        raise ValueError("per_label must be positive")
    invalid_modes = sorted(set(prompt_modes) - set(DEFAULT_PROMPTS))
    if invalid_modes:
        raise ValueError(f"unsupported prompt modes: {invalid_modes}")
    if not prompt_modes:
        raise ValueError("at least one prompt mode is required")

    catalog = Path(catalog_path).expanduser().resolve()
    root = Path(audio_root).expanduser().resolve()
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in read_source_catalog(catalog):
        if record.duration_sec is None or record.file_sha256 is None:
            raise ValueError(
                f"catalog record {record.source_id} has not been probed; use prepared/all.jsonl"
            )
        label = record.labels[0] if record.labels else record.kind
        grouped[label].append(record)

    selected: list[Any] = []
    for _label, records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda item: _stable_key(seed, item.source_id))
        selected.extend(ordered[:per_label])
    selected.sort(key=lambda item: (item.labels[0] if item.labels else item.kind, item.source_id))

    items: list[dict[str, Any]] = []
    for record in selected:
        audio = (root / record.audio_path).resolve()
        try:
            audio.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"audio path escapes root: {record.audio_path}") from exc
        if not audio.is_file():
            raise FileNotFoundError(audio)
        observed_hash = file_sha256(audio)
        if observed_hash != record.file_sha256:
            raise ValueError(f"audio changed after catalog preparation: {record.source_id}")
        items.append(
            {
                "source_id": record.source_id,
                "kind": record.kind,
                "label": record.labels[0] if record.labels else record.kind,
                "audio_path": str(audio),
                "duration_sec": record.duration_sec,
                "audio_sha256": observed_hash,
                "split": record.split,
            }
        )

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "catalog_path": str(catalog),
        "catalog_sha256": file_sha256(catalog),
        "audio_root": str(root),
        "seed": seed,
        "per_label": per_label,
        "prompt_modes": list(prompt_modes),
        "prompts": {mode: DEFAULT_PROMPTS[mode] for mode in prompt_modes},
        "n_labels": len(grouped),
        "n_sources": len(items),
        "n_generations": len(items) * len(prompt_modes),
        "items": items,
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"frozen plan already exists; choose another path: {destination}")
    _write_json(destination, plan)
    return plan


def _parse_structured(raw_text: str, sample_id: str, duration: float) -> tuple[bool, int]:
    try:
        ledger = atomic_to_ledger(raw_text, sample_id, duration)
        if ledger.events:
            return True, len(ledger.events)
    except Exception:
        pass
    ledger, report = parse_model_output(raw_text, sample_id=sample_id, duration_sec=duration)
    return report.strict_format_success, len(ledger.events)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL {path}:{line_no}") from exc
    return rows


def run_source_caption_plan(
    plan_path: str | Path,
    output_path: str | Path,
    adapter: AudioCaptioner,
    *,
    resume: bool = False,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every frozen source/prompt pair and persist each result immediately."""
    plan_file = Path(plan_path).expanduser().resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"unsupported plan schema: {plan.get('schema_version')}")
    plan_hash = file_sha256(plan_file)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(output) if resume else []
    if output.exists() and not resume:
        raise FileExistsError(f"output already exists; use --resume or choose another path: {output}")
    for row in existing:
        if row.get("plan_sha256") != plan_hash:
            raise ValueError("existing results belong to a different frozen plan")
    existing_keys = [(row["source_id"], row["prompt_mode"]) for row in existing]
    if len(existing_keys) != len(set(existing_keys)):
        raise ValueError("existing results contain duplicate source/prompt pairs")
    successful_existing = [row for row in existing if not row.get("error")]
    completed = {
        (row["source_id"], row["prompt_mode"]) for row in successful_existing
    }
    expected = {
        (item["source_id"], mode)
        for item in plan["items"]
        for mode in plan["prompt_modes"]
    }
    if not completed <= expected:
        raise ValueError("existing results contain source/prompt pairs not present in the plan")

    # In resume mode successful rows are preserved and errored rows are retried.
    # Clean failed rows transactionally before appending, avoiding duplicate keys.
    if resume and output.exists() and len(successful_existing) != len(existing):
        temporary = output.with_suffix(output.suffix + ".resume.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in successful_existing:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(output)
    file_mode = "a" if resume and output.exists() else "w"
    with output.open(file_mode, encoding="utf-8", newline="\n") as handle:
        for item in plan["items"]:
            audio = Path(item["audio_path"])
            if file_sha256(audio) != item["audio_sha256"]:
                raise ValueError(f"planned audio changed: {item['source_id']}")
            for prompt_mode in plan["prompt_modes"]:
                key = (item["source_id"], prompt_mode)
                if key in completed:
                    continue
                started = time.perf_counter()
                raw_text = ""
                error = None
                try:
                    raw_text = adapter.infer(
                        str(audio),
                        plan["prompts"][prompt_mode],
                        sample_id=item["source_id"],
                        duration=float(item["duration_sec"]),
                    )
                except Exception as exc:  # retain partial experiment instead of losing GPU hours
                    error = f"{type(exc).__name__}: {exc}"
                parseable, event_count = (
                    _parse_structured(raw_text, item["source_id"], float(item["duration_sec"]))
                    if prompt_mode == "structured" and raw_text
                    else (False, 0)
                )
                result = {
                    "schema_version": RESULT_SCHEMA,
                    "plan_sha256": plan_hash,
                    "source_id": item["source_id"],
                    "kind": item["kind"],
                    "label": item["label"],
                    "audio_path": item["audio_path"],
                    "audio_sha256": item["audio_sha256"],
                    "duration_sec": item["duration_sec"],
                    "prompt_mode": prompt_mode,
                    "prompt": plan["prompts"][prompt_mode],
                    "raw_text": raw_text,
                    "error": error,
                    "elapsed_sec": round(time.perf_counter() - started, 4),
                    "format_parseable_auto": parseable,
                    "parsed_event_count": event_count,
                    "model": model_metadata or {},
                }
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                completed.add(key)

    rows = _read_jsonl(output)
    errors = [row for row in rows if row.get("error")]
    return {
        "schema_version": RESULT_SCHEMA,
        "pass": len(rows) == len(expected) and not errors,
        "plan_path": str(plan_file),
        "plan_sha256": plan_hash,
        "output_path": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "n_expected": len(expected),
        "n_results": len(rows),
        "n_errors": len(errors),
        "n_structured_parseable": sum(
            row["prompt_mode"] == "structured" and row["format_parseable_auto"] for row in rows
        ),
    }


def write_source_caption_audit(results_path: str | Path, output_path: str | Path) -> int:
    """Materialize a review sheet; the human must listen, not judge text alone."""
    rows = _read_jsonl(Path(results_path))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"audit sheet already exists; refusing to overwrite review: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["label"], item["source_id"], item["prompt_mode"])):
            writer.writerow(
                {
                    "source_id": row["source_id"],
                    "label": row["label"],
                    "kind": row["kind"],
                    "prompt_mode": row["prompt_mode"],
                    "audio_path": row["audio_path"],
                    "raw_text": row["raw_text"],
                    "error": row.get("error") or "",
                    "parsed_event_count": row["parsed_event_count"],
                    "format_parseable_auto": "y" if row["format_parseable_auto"] else "n",
                    "structured_format_usable_y_n_or_na": (
                        "" if row["prompt_mode"] == "structured" else "na"
                    ),
                    "temporal_claims_supported_y_n_or_na": "",
                }
            )
    return len(rows)


def _answer(row: dict[str, str], field: str, allowed: set[str]) -> str:
    value = str(row.get(field, "")).strip().lower()
    if value not in allowed:
        raise ValueError(f"{row.get('source_id')}:{row.get('prompt_mode')} {field}={value!r}")
    return value


def validate_source_caption_audit(
    plan_path: str | Path,
    results_path: str | Path,
    audit_path: str | Path,
    output_path: str | Path,
    *,
    min_label_correct: float = 0.70,
    min_coverage: float = 0.70,
    min_hallucination_free: float = 0.80,
    min_structured_format: float = 0.50,
    min_temporal_supported: float = 0.70,
    min_structured_temporal_claims: float = 0.50,
) -> dict[str, Any]:
    """Bind human answers to exact generations and produce a go/no-go decision."""
    plan_file, results_file, audit_file = map(Path, (plan_path, results_path, audit_path))
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    results = _read_jsonl(results_file)
    result_keys = [(row["source_id"], row["prompt_mode"]) for row in results]
    result_by_key = {
        (row["source_id"], row["prompt_mode"]): row for row in results
    }
    expected = {
        (item["source_id"], mode)
        for item in plan["items"]
        for mode in plan["prompt_modes"]
    }
    with audit_file.open("r", encoding="utf-8-sig", newline="") as handle:
        audit = list(csv.DictReader(handle))
    audit_keys = [(row["source_id"], row["prompt_mode"]) for row in audit]
    audit_by_key = {
        (row["source_id"], row["prompt_mode"]): row for row in audit
    }

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    check("results_unique", len(result_keys) == len(set(result_keys)), len(result_keys) - len(set(result_keys)))
    check("audit_unique", len(audit_keys) == len(set(audit_keys)), len(audit_keys) - len(set(audit_keys)))
    check("results_complete", set(result_by_key) == expected, {"expected": len(expected), "observed": len(result_by_key)})
    check("audit_complete", set(audit_by_key) == expected, {"expected": len(expected), "observed": len(audit_by_key)})
    current_plan_hash = file_sha256(plan_file)
    wrong_plan = [key for key, row in result_by_key.items() if row.get("plan_sha256") != current_plan_hash]
    check("results_bound_to_plan", not wrong_plan, wrong_plan[:20])
    check("inference_error_free", not any(row.get("error") for row in results), [row["source_id"] for row in results if row.get("error")][:20])
    text_mismatch = [key for key in expected if key in result_by_key and key in audit_by_key and audit_by_key[key].get("raw_text") != result_by_key[key].get("raw_text")]
    check("audit_bound_to_raw_text", not text_mismatch, text_mismatch[:20])

    invalid: list[str] = []
    scored: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    temporal_claims: dict[str, list[bool]] = defaultdict(list)
    for key, row in audit_by_key.items():
        mode = key[1]
        try:
            scored[mode]["label_correct"].append(_answer(row, "label_correct_y_n", {"y", "n"}) == "y")
            scored[mode]["coverage"].append(_answer(row, "all_audible_events_covered_y_n", {"y", "n"}) == "y")
            scored[mode]["hallucination_free"].append(_answer(row, "hallucination_free_y_n", {"y", "n"}) == "y")
            has_time = _answer(row, "temporal_claims_present_y_n", {"y", "n"}) == "y"
            temporal_claims[mode].append(has_time)
            temporal = _answer(row, "temporal_claims_supported_y_n_or_na", {"y", "n"} if has_time else {"na"})
            if has_time:
                scored[mode]["temporal_supported"].append(temporal == "y")
            structured = _answer(
                row,
                "structured_format_usable_y_n_or_na",
                {"y", "n"} if mode == "structured" else {"na"},
            )
            if mode == "structured":
                scored[mode]["structured_format"].append(structured == "y")
        except ValueError as exc:
            invalid.append(str(exc))
    check("human_answers_valid", not invalid, invalid[:20])

    rates: dict[str, dict[str, float | None]] = {}
    for mode in plan["prompt_modes"]:
        rates[mode] = {}
        for metric, values in scored[mode].items():
            rates[mode][metric] = round(sum(values) / len(values), 6) if values else None
        claims = temporal_claims[mode]
        rates[mode]["temporal_claim_rate"] = round(sum(claims) / len(claims), 6) if claims else None
        for metric, minimum in (
            ("label_correct", min_label_correct),
            ("coverage", min_coverage),
            ("hallucination_free", min_hallucination_free),
        ):
            value = rates[mode].get(metric)
            check(f"{mode}_{metric}", value is not None and value >= minimum, {"observed": value, "minimum": minimum})
    if "structured" in plan["prompt_modes"]:
        structured_results = [row for row in results if row["prompt_mode"] == "structured"]
        rates["structured"]["auto_parseable"] = (
            round(
                sum(bool(row.get("format_parseable_auto")) for row in structured_results)
                / len(structured_results),
                6,
            )
            if structured_results
            else None
        )
    structured_rate = rates.get("structured", {}).get("structured_format")
    if "structured" in plan["prompt_modes"]:
        check("structured_format", structured_rate is not None and structured_rate >= min_structured_format, {"observed": structured_rate, "minimum": min_structured_format})
        auto_rate = rates["structured"].get("auto_parseable")
        check("structured_auto_parseable", auto_rate is not None and auto_rate >= min_structured_format, {"observed": auto_rate, "minimum": min_structured_format})
        claim_rate = rates["structured"].get("temporal_claim_rate")
        check("structured_temporal_claim_rate", claim_rate is not None and claim_rate >= min_structured_temporal_claims, {"observed": claim_rate, "minimum": min_structured_temporal_claims})
        temporal_rate = rates["structured"].get("temporal_supported")
        check("structured_temporal_supported", temporal_rate is not None and temporal_rate >= min_temporal_supported, {"observed": temporal_rate, "minimum": min_temporal_supported})

    report = {
        "schema_version": AUDIT_SCHEMA,
        "pass": all(item["pass"] for item in checks),
        "plan_sha256": file_sha256(plan_file),
        "results_sha256": file_sha256(results_file),
        "audit_sha256": file_sha256(audit_file),
        "n_expected": len(expected),
        "rates": rates,
        "thresholds": {
            "label_correct": min_label_correct,
            "coverage": min_coverage,
            "hallucination_free": min_hallucination_free,
            "structured_format": min_structured_format,
            "temporal_supported": min_temporal_supported,
            "structured_temporal_claim_rate": min_structured_temporal_claims,
        },
        "checks": checks,
    }
    _write_json(Path(output_path), report)
    return report


__all__ = [
    "AUDIT_FIELDS",
    "DEFAULT_PROMPTS",
    "build_source_caption_plan",
    "run_source_caption_plan",
    "validate_source_caption_audit",
    "write_source_caption_audit",
]
