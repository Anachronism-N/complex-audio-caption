"""Top-level evaluation: aggregate per-sample metrics over a corpus.

Reads prediction/reference JSONL (one :class:`Ledger` per line, ``sample_id``
join key) and reports:

* strict format-success rate (from parser reports, when predictions are raw
  model output) — when predictions are already Ledgers this is 1.0;
* semantic-temporal event F1, precision, recall;
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
    mean_matched_text_similarity: float = 0.0
    zero_text_match_rate: float = 0.0
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    strict_format_success: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorpusMetrics:
    n_samples: int
    tiou_threshold: float
    min_text_similarity: float
    strict_format_success_rate: float
    macro_event_precision: float
    macro_event_recall: float
    macro_event_f1: float
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
    mean_matched_text_similarity: float
    macro_zero_text_match_rate: float
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
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return out


def evaluate_sample(
    ref: Ledger,
    hyp: Ledger,
    strict_format_success: bool = True,
    *,
    tiou_threshold: float = 0.3,
    min_text_similarity: float = 0.0,
    warnings: list[str] | None = None,
) -> SampleMetrics:
    matches = match_events(
        ref.events,
        hyp.events,
        tiou_threshold=tiou_threshold,
        min_text_similarity=min_text_similarity,
    )
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
    text_scores = [match.text_sim for match in pointer_matches]
    mean_text_similarity = (
        sum(text_scores) / len(text_scores) if text_scores else 0.0
    )
    zero_text_match_rate = (
        sum(score == 0.0 for score in text_scores) / len(text_scores)
        if text_scores
        else 0.0
    )

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
        mean_matched_text_similarity=round(mean_text_similarity, 6),
        zero_text_match_rate=round(zero_text_match_rate, 6),
        per_type=_per_type_breakdown(matches),
        strict_format_success=strict_format_success,
        warnings=list(warnings or []),
    )


def _macro(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def aggregate(
    samples: list[SampleMetrics],
    *,
    tiou_threshold: float = 0.3,
    min_text_similarity: float = 0.0,
) -> CorpusMetrics:
    n = len(samples)
    if n == 0:
        return CorpusMetrics(
            n_samples=0,
            tiou_threshold=tiou_threshold,
            min_text_similarity=min_text_similarity,
            strict_format_success_rate=0.0,
            macro_event_precision=0.0,
            macro_event_recall=0.0,
            macro_event_f1=0.0,
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
            mean_matched_text_similarity=0.0,
            macro_zero_text_match_rate=0.0,
        )

    # Micro-aggregate per type.  Averaging only samples where a type appears
    # hides complete omissions; accumulating TP/FP/FN does not.
    types = sorted({t for s in samples for t in s.per_type})
    macro_per_type: dict[str, dict[str, float]] = {}
    for t in types:
        rows = [s.per_type[t] for s in samples if t in s.per_type]
        tp = sum(row.get("tp", 0.0) for row in rows)
        fp = sum(row.get("fp", 0.0) for row in rows)
        fn = sum(row.get("fn", 0.0) for row in rows)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_per_type[t] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    return CorpusMetrics(
        n_samples=n,
        tiou_threshold=tiou_threshold,
        min_text_similarity=min_text_similarity,
        strict_format_success_rate=_macro(
            [1.0 if s.strict_format_success else 0.0 for s in samples]
        ),
        macro_event_precision=_macro([s.event_precision for s in samples]),
        macro_event_recall=_macro([s.event_recall for s in samples]),
        macro_event_f1=_macro([s.event_f1 for s in samples]),
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
        mean_matched_text_similarity=_macro(
            [s.mean_matched_text_similarity for s in samples]
        ),
        macro_zero_text_match_rate=_macro([s.zero_text_match_rate for s in samples]),
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


def evaluate_corpus(
    predictions: str | Path | dict[str, Ledger],
    references: str | Path | dict[str, Ledger],
    *,
    parse_reports: str | Path | dict[str, dict] | None = None,
    tiou_threshold: float = 0.3,
    min_text_similarity: float = 0.0,
) -> CorpusMetrics:
    """Evaluate predictions against references.

    Both inputs may be a path to a JSONL file (one Ledger per line) or a
    ``{sample_id: Ledger}`` mapping. Predictions and references are joined on
    ``sample_id``.
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
    if isinstance(parse_reports, (str, Path)):
        report_payload = json.loads(Path(parse_reports).read_text(encoding="utf-8"))
        report_map = {
            row["sample_id"]: row for row in report_payload.get("samples", [])
        }
    else:
        report_map = dict(parse_reports or {})

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
                mean_matched_text_similarity=0.0,
                zero_text_match_rate=0.0,
                strict_format_success=False,
                warnings=["prediction missing"],
            )
        else:
            parse_report = report_map.get(sid)
            strict = (
                bool(parse_report.get("strict_format_success", False))
                if parse_report is not None
                else True
            )
            sm = evaluate_sample(
                ref,
                hyp,
                strict_format_success=strict,
                tiou_threshold=tiou_threshold,
                min_text_similarity=min_text_similarity,
                warnings=list(parse_report.get("warnings", []))
                if parse_report is not None
                else None,
            )
        samples.append(sm)
    return aggregate(
        samples,
        tiou_threshold=tiou_threshold,
        min_text_similarity=min_text_similarity,
    )


__all__ = [
    "CorpusMetrics",
    "SampleMetrics",
    "aggregate",
    "evaluate_corpus",
    "evaluate_sample",
]
