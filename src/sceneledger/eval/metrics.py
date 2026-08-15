"""Top-level evaluation: aggregate per-sample metrics over a corpus.

Reads prediction/reference JSONL (one :class:`Ledger` per line, ``sample_id``
join key) and reports:

* strict format-success rate, read from the inference parser report.  A
  parsed Ledger alone cannot prove that the raw model output was valid, so
  the rate is reported as unavailable when that evidence is missing;
* type/temporal event F1, precision, recall;
* lexical caption token-F1 over temporally matched events, with omissions
  contributing zero instead of disappearing from the score;
* onset/offset MAE and p90, tolerance accuracy at several collars;
* per-type breakdown;
* hallucination / omission counts;
* source-count MAE and pointer accuracy;
* frame-level SegF1@100ms.

The CLI in :mod:`sceneledger.cli.evaluate` wraps :func:`evaluate_corpus`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sceneledger.data.schema import Ledger
from sceneledger.eval.event_matcher import EventMatch, match_events, matched_pairs
from sceneledger.eval.temporal import (
    BoundaryErrors,
    boundary_mae,
    seg_f1,
    tolerance_accuracy,
)


@dataclass
class SampleMetrics:
    sample_id: str
    n_ref: int
    n_hyp: int
    n_matched: int
    event_precision: float
    event_recall: float
    event_f1: float
    seg_f1_100ms: float
    onset_mae: float
    offset_mae: float
    onset_p90: float
    offset_p90: float
    tolerance_acc_010: float
    tolerance_acc_025: float
    tolerance_acc_050: float
    tolerance_acc_100: float
    hallucination: int  # hyp events with no match
    omission: int  # ref events with no match
    source_count_mae: float
    pointer_accuracy: float
    # Defaults to zero so historical metrics JSON can still be loaded by the
    # robustness reporter; new evaluation runs always set it explicitly.
    caption_token_f1: float = 0.0
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    strict_format_success: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorpusMetrics:
    n_samples: int
    strict_format_success_rate: float | None
    format_status_complete: bool
    n_format_status_known: int
    n_format_status_missing: int
    macro_event_precision: float
    macro_event_recall: float
    macro_event_f1: float
    macro_caption_token_f1: float
    macro_seg_f1_100ms: float
    mean_onset_mae: float
    mean_offset_mae: float
    mean_onset_p90: float
    mean_offset_p90: float
    macro_tolerance_acc_010: float
    macro_tolerance_acc_025: float
    macro_tolerance_acc_050: float
    macro_tolerance_acc_100: float
    total_hallucination: int
    total_omission: int
    mean_source_count_mae: float
    mean_pointer_accuracy: float
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
def _per_type_breakdown(matches: list[EventMatch]) -> dict[str, dict[str, float]]:
    types = sorted({m.type for m in matches if m.type is not None})
    out: dict[str, dict[str, float]] = {}
    for t in types:
        tp = sum(1 for m in matches if m.type == t and m.is_match)
        # false alarms: hyps of this type with no match
        fp = sum(1 for m in matches if m.type == t and m.ref_id is None)
        fn = sum(1 for m in matches if m.type == t and m.hyp_id is None)
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[t] = {  # type: ignore[assignment]
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "caption_token_f1": round(
                sum(m.text_sim for m in matches if m.type == t and m.is_match)
                / max(1, tp + fn),
                6,
            ),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return out


def evaluate_sample(
    ref: Ledger,
    hyp: Ledger,
    strict_format_success: bool | None = None,
    warnings: list[str] | None = None,
) -> SampleMetrics:
    matches = match_events(ref.events, hyp.events)
    pairs = matched_pairs(matches, ref.events, hyp.events)

    n_ref = len(ref.events)
    n_hyp = len(hyp.events)
    n_matched = sum(1 for m in matches if m.is_match)

    precision = n_matched / n_hyp if n_hyp else 1.0
    recall = n_matched / n_ref if n_ref else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    # Event validity is deliberately defined by type + temporal overlap in the
    # matcher.  Report caption quality separately so a structurally correct but
    # semantically wrong prediction cannot be advertised as caption success.
    # Reference omissions contribute zero; hallucinations remain visible in
    # event precision/hallucination counts.
    caption_token_f1 = (
        sum(match.text_sim for match in matches if match.is_match) / n_ref
        if n_ref
        else (1.0 if n_hyp == 0 else 0.0)
    )

    _, _, segf1 = seg_f1(ref.events, hyp.events, collar_seconds=0.1)

    berr: BoundaryErrors = boundary_mae(pairs)

    ref_count = len(ref.tracks)
    hyp_count = len(hyp.tracks)
    source_count_mae = abs(ref_count - hyp_count)

    # pointer accuracy: among matched pairs, fraction with agreeing track_id
    pointer_matches = [m for m in matches if m.is_match]
    if pointer_matches:
        pointer_accuracy = sum(1 for m in pointer_matches if m.track_match) / len(
            pointer_matches
        )
    else:
        pointer_accuracy = 1.0

    hallucination = sum(1 for m in matches if m.ref_id is None)
    omission = sum(1 for m in matches if m.hyp_id is None)

    return SampleMetrics(
        sample_id=ref.sample_id,
        n_ref=n_ref,
        n_hyp=n_hyp,
        n_matched=n_matched,
        event_precision=round(precision, 6),
        event_recall=round(recall, 6),
        event_f1=round(f1, 6),
        caption_token_f1=round(caption_token_f1, 6),
        seg_f1_100ms=round(segf1, 6),
        onset_mae=berr.onset_mae,
        offset_mae=berr.offset_mae,
        onset_p90=berr.onset_p90,
        offset_p90=berr.offset_p90,
        tolerance_acc_010=tolerance_accuracy(pairs, 0.1),
        tolerance_acc_025=tolerance_accuracy(pairs, 0.25),
        tolerance_acc_050=tolerance_accuracy(pairs, 0.5),
        tolerance_acc_100=tolerance_accuracy(pairs, 1.0),
        hallucination=hallucination,
        omission=omission,
        source_count_mae=round(float(source_count_mae), 6),
        pointer_accuracy=round(pointer_accuracy, 6),
        per_type=_per_type_breakdown(matches),
        strict_format_success=strict_format_success,
        warnings=list(warnings or []),
    )


def _macro(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def aggregate(samples: list[SampleMetrics]) -> CorpusMetrics:
    n = len(samples)
    if n == 0:
        return CorpusMetrics(
            n_samples=0,
            strict_format_success_rate=None,
            format_status_complete=False,
            n_format_status_known=0,
            n_format_status_missing=0,
            macro_event_precision=0.0,
            macro_event_recall=0.0,
            macro_event_f1=0.0,
            macro_caption_token_f1=0.0,
            macro_seg_f1_100ms=0.0,
            mean_onset_mae=0.0,
            mean_offset_mae=0.0,
            mean_onset_p90=0.0,
            mean_offset_p90=0.0,
            macro_tolerance_acc_010=0.0,
            macro_tolerance_acc_025=0.0,
            macro_tolerance_acc_050=0.0,
            macro_tolerance_acc_100=0.0,
            total_hallucination=0,
            total_omission=0,
            mean_source_count_mae=0.0,
            mean_pointer_accuracy=0.0,
        )

    # macro per-type
    types = sorted({t for s in samples for t in s.per_type})
    macro_per_type: dict[str, dict[str, float]] = {}
    for t in types:
        rows = [s.per_type[t] for s in samples if t in s.per_type]
        macro_per_type[t] = {
            k: _macro([r[k] for r in rows])
            for k in ("precision", "recall", "f1", "caption_token_f1")
        }

    known_format = [s.strict_format_success for s in samples if s.strict_format_success is not None]
    format_complete = len(known_format) == n

    return CorpusMetrics(
        n_samples=n,
        strict_format_success_rate=(
            _macro([1.0 if status else 0.0 for status in known_format])
            if format_complete
            else None
        ),
        format_status_complete=format_complete,
        n_format_status_known=len(known_format),
        n_format_status_missing=n - len(known_format),
        macro_event_precision=_macro([s.event_precision for s in samples]),
        macro_event_recall=_macro([s.event_recall for s in samples]),
        macro_event_f1=_macro([s.event_f1 for s in samples]),
        macro_caption_token_f1=_macro([s.caption_token_f1 for s in samples]),
        macro_seg_f1_100ms=_macro([s.seg_f1_100ms for s in samples]),
        mean_onset_mae=_macro([s.onset_mae for s in samples]),
        mean_offset_mae=_macro([s.offset_mae for s in samples]),
        mean_onset_p90=_macro([s.onset_p90 for s in samples]),
        mean_offset_p90=_macro([s.offset_p90 for s in samples]),
        macro_tolerance_acc_010=_macro([s.tolerance_acc_010 for s in samples]),
        macro_tolerance_acc_025=_macro([s.tolerance_acc_025 for s in samples]),
        macro_tolerance_acc_050=_macro([s.tolerance_acc_050 for s in samples]),
        macro_tolerance_acc_100=_macro([s.tolerance_acc_100 for s in samples]),
        total_hallucination=sum(s.hallucination for s in samples),
        total_omission=sum(s.omission for s in samples),
        mean_source_count_mae=_macro([s.source_count_mae for s in samples]),
        mean_pointer_accuracy=_macro([s.pointer_accuracy for s in samples]),
        per_type=macro_per_type,
        samples=[s.to_dict() for s in samples],
    )


# --------------------------------------------------------------------------- #
def _load_ledger_jsonl(path: str | Path) -> dict[str, Ledger]:
    out: dict[str, Ledger] = {}
    p = Path(path)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # accept either a bare Ledger or a manifest entry with a target_ledger field
        if "target_ledger" in obj and "schema_version" not in obj:
            obj = obj["target_ledger"]
        ledger = Ledger.model_validate(obj)
        out[ledger.sample_id] = ledger
    return out


def load_inference_report(source: str | Path | dict) -> tuple[dict, dict[str, dict]]:
    """Load and validate per-sample parser evidence from inference.

    The returned mapping contains only validated ``strict_format_success``
    booleans and warning strings.  Duplicate/missing IDs and internally
    inconsistent summary fields are rejected instead of silently producing a
    paper metric from incomplete evidence.
    """
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = dict(source)
    if not isinstance(payload, dict):
        raise ValueError("inference report must be a JSON object")
    rows = payload.get("samples")
    if not isinstance(rows, list):
        raise ValueError("inference report is missing a samples list")
    if payload.get("n_samples") is not None and payload["n_samples"] != len(rows):
        raise ValueError("inference report n_samples does not match its samples list")

    statuses: dict[str, dict] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"inference report sample {index} must be an object")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"inference report sample {index} has no sample_id")
        if sample_id in statuses:
            raise ValueError(f"duplicate sample_id in inference report: {sample_id}")
        status = row.get("strict_format_success")
        if not isinstance(status, bool):
            raise ValueError(
                f"inference report sample {sample_id} has no boolean "
                "strict_format_success"
            )
        warnings = row.get("warnings", [])
        if not isinstance(warnings, list) or not all(
            isinstance(item, str) for item in warnings
        ):
            raise ValueError(f"inference report sample {sample_id} has invalid warnings")
        statuses[sample_id] = {
            "strict_format_success": status,
            "warnings": list(warnings),
        }

    reported_rate = payload.get("strict_format_success_rate")
    computed_rate = round(
        sum(1 for row in statuses.values() if row["strict_format_success"])
        / max(1, len(statuses)),
        4,
    )
    if reported_rate is not None:
        if not isinstance(reported_rate, (int, float)) or round(
            float(reported_rate), 4
        ) != computed_rate:
            raise ValueError(
                "inference report strict_format_success_rate is inconsistent "
                "with its samples"
            )
    reported_count = payload.get("n_strict_format_success")
    computed_count = sum(
        1 for row in statuses.values() if row["strict_format_success"]
    )
    if reported_count is not None and reported_count != computed_count:
        raise ValueError(
            "inference report n_strict_format_success is inconsistent with its samples"
        )
    return payload, statuses


def evaluate_corpus(
    predictions: str | Path | dict[str, Ledger],
    references: str | Path | dict[str, Ledger],
    inference_report: str | Path | dict | None = None,
) -> CorpusMetrics:
    """Evaluate predictions against references.

    Both inputs may be a path to a JSONL file (one Ledger per line) or a
    ``{sample_id: Ledger}`` mapping. Predictions and references are joined on
    ``sample_id``. ``inference_report`` is the raw-output parser report emitted
    by :mod:`sceneledger.cli.infer`; without it, format success is unknown and
    is never inferred from the existence of a Ledger.
    """
    refs = (
        _load_ledger_jsonl(references)
        if isinstance(references, (str, Path))
        else dict(references)
    )
    hyps = (
        _load_ledger_jsonl(predictions)
        if isinstance(predictions, (str, Path))
        else dict(predictions)
    )
    format_statuses: dict[str, dict] | None = None
    if inference_report is not None:
        _, format_statuses = load_inference_report(inference_report)
        reference_ids = set(refs)
        report_ids = set(format_statuses)
        if report_ids != reference_ids:
            missing = sorted(reference_ids - report_ids)[:5]
            extra = sorted(report_ids - reference_ids)[:5]
            raise ValueError(
                "inference report sample IDs do not equal reference IDs: "
                f"missing={missing}, extra={extra}"
            )

    samples: list[SampleMetrics] = []
    for sid in sorted(refs):
        ref = refs[sid]
        hyp = hyps.get(sid)
        if hyp is None:
            sm = SampleMetrics(
                sample_id=sid,
                n_ref=len(ref.events),
                n_hyp=0,
                n_matched=0,
                event_precision=0.0,
                event_recall=0.0,
                event_f1=0.0,
                seg_f1_100ms=0.0,
                onset_mae=0.0,
                offset_mae=0.0,
                onset_p90=0.0,
                offset_p90=0.0,
                tolerance_acc_010=0.0,
                tolerance_acc_025=0.0,
                tolerance_acc_050=0.0,
                tolerance_acc_100=0.0,
                hallucination=0,
                omission=len(ref.events),
                source_count_mae=float(len(ref.tracks)),
                pointer_accuracy=0.0,
                strict_format_success=False,
                warnings=["prediction missing"],
            )
        else:
            evidence = format_statuses.get(sid) if format_statuses is not None else None
            sm = evaluate_sample(
                ref,
                hyp,
                strict_format_success=(
                    evidence["strict_format_success"] if evidence is not None else None
                ),
                warnings=evidence["warnings"] if evidence is not None else None,
            )
        samples.append(sm)
    return aggregate(samples)


__all__ = [
    "CorpusMetrics",
    "SampleMetrics",
    "aggregate",
    "evaluate_corpus",
    "evaluate_sample",
    "load_inference_report",
]
