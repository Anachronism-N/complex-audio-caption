"""Tolerant parsing of raw model output into a :class:`Ledger`.

The strict :func:`sceneledger.models.serializer.deserialize` expects a clean
XML document. Real model output is messier: surrounding prose, missing
``<ledger>`` root, missing tracks header, missing ``confidence``, lowercase
booleans, smart quotes, etc. This module extracts whatever typed events it
can and records a :class:`ParseReport` describing what survived and what was
rejected. Evaluation reports both strict format-success rate (the fraction of
samples whose output parsed without any recovery) and recovered event counts.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from sceneledger.data.schema import (
    SCHEMA_VERSION,
    TIME_RESOLUTION_SEC,
    Event,
    Ledger,
    Span,
    Track,
)
from sceneledger.models.serializer import _make_span_objects, _parse_span_list

_EVENT_TAGS = ("speech", "lys", "music", "sfx")
_TAG_RE = re.compile(
    r"<(?P<tag>speech|lys|music|sfx)\b(?P<attrs>[^>]*)>(?P<text>.*?)</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)
_ATTR_RE = re.compile(r'(?P<k>[\w-]+)\s*=\s*"(?P<v>[^"]*)"')
_BARE_TAG_RE = re.compile(
    r"<(?P<tag>speech|lys|music|sfx)\b([^>]*)/>", re.IGNORECASE
)

_BOOL_TRUE = {"true", "1", "yes", "y"}


@dataclass
class RecoveredEvent:
    raw_tag: str
    raw_attrs: dict[str, str]
    text: str
    error: str | None = None  # set if partially recovered


@dataclass
class ParseReport:
    sample_id: str
    ok: bool  # True iff parsed with zero recoveries / warnings
    events_recovered: int = 0
    events_rejected: int = 0
    warnings: list[str] = field(default_factory=list)
    rejected_snippets: list[str] = field(default_factory=list)
    strict_format_success: bool = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        self.ok = False


# --------------------------------------------------------------------------- #
# attribute coercion
# --------------------------------------------------------------------------- #
def _parse_attrs(attr_string: str) -> dict[str, str]:
    return {m.group("k"): m.group("v") for m in _ATTR_RE.finditer(attr_string)}


def _coerce_confidence(attrs: dict[str, str], report: ParseReport, eid: str) -> float:
    raw = attrs.get("confidence")
    if raw is None:
        report.add_warning(f"event {eid} missing confidence, defaulting to 1.0")
        return 1.0
    try:
        return float(raw)
    except ValueError:
        report.add_warning(f"event {eid} confidence={raw!r} not numeric, defaulting 1.0")
        return 1.0


def _coerce_spans(
    attrs: dict[str, str], report: ParseReport, eid: str
) -> list[Span]:
    raw_t = attrs.get("t") or attrs.get("time") or attrs.get("spans")
    if raw_t is None:
        raise ValueError(f"event {eid} missing time span ('t' attribute)")
    spans = _parse_span_list(raw_t)
    if not spans:
        raise ValueError(f"event {eid} has empty span list {raw_t!r}")
    return _make_span_objects(spans)


def _coerce_id(attrs: dict[str, str], idx: int) -> str:
    eid = attrs.get("id")
    if eid is not None and re.match(r"^E[0-9]+$", eid):
        return eid
    # assign a stable synthetic id
    return f"E{idx:03d}"


def _coerce_track_id(attrs: dict[str, str]) -> str | None:
    tid = attrs.get("track") or attrs.get("track_id")
    if tid is None:
        return None
    if re.match(r"^T[0-9]+$", tid):
        return tid
    return None  # malformed track refs are dropped (not fatal)


def _coerce_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    return raw.strip().lower() in _BOOL_TRUE


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def parse_model_output(
    text: str, sample_id: str, duration_sec: float = 30.0
) -> tuple[Ledger, ParseReport]:
    """Parse raw model output into a :class:`Ledger` + :class:`ParseReport`.

    Strategy:
    1. Try strict :func:`deserialize` (fast path, sets strict_format_success).
    2. Otherwise, regex-scan for ``<speech|lys|music|sfx ...>text</...>`` and
       self-closed variants; coerce attributes leniently.
    3. Infer tracks from recovered events' track_ids (kind derived from event
       type when possible).
    """
    report = ParseReport(sample_id=sample_id, ok=True)

    # 1. strict fast path
    try:
        ledger = _try_strict(text, sample_id, duration_sec)
        report.strict_format_success = True
        report.events_recovered = len(ledger.events)
        return ledger, report
    except (ET.ParseError, ValueError) as exc:
        # fall through to tolerant path; record only if it looked like XML-ish
        report.ok = False
        report.strict_format_success = False
        if "<" in text:
            report.warnings.append(f"strict parse failed: {exc}")

    # 2. tolerant regex scan
    recovered: list[RecoveredEvent] = []

    for m in _TAG_RE.finditer(text):
        tag = m.group("tag").lower()
        attrs = _parse_attrs(m.group("attrs"))
        text_body = m.group("text").strip()
        recovered.append(RecoveredEvent(raw_tag=tag, raw_attrs=attrs, text=text_body))

    for m in _BARE_TAG_RE.finditer(text):
        tag = m.group("tag").lower()
        attrs = _parse_attrs(m.group(2))
        recovered.append(RecoveredEvent(raw_tag=tag, raw_attrs=attrs, text=""))

    # de-duplicate overlapping matches (paired tags already cover self-closing
    # forms written as <tag .../> because the paired regex requires a close)
    seen_spans = set()
    events: list[Event] = []
    used_ids: set[str] = set()
    for i, rec in enumerate(recovered):
        eid = _coerce_id(rec.raw_attrs, i)
        if eid in used_ids:
            eid = f"E{i:03d}"
        used_ids.add(eid)
        try:
            spans = _coerce_spans(rec.raw_attrs, report, eid)
            if not rec.text:
                raise ValueError("empty event text")
            event = Event(
                id=eid,
                type=rec.raw_tag,  # type: ignore[arg-type]
                track_id=_coerce_track_id(rec.raw_attrs),
                spans=spans,
                text=rec.text,
                verbatim=_coerce_bool(rec.raw_attrs.get("verbatim")),
                language=rec.raw_attrs.get("language"),
                confidence=_coerce_confidence(rec.raw_attrs, report, eid),
            )
            events.append(event)
            key = (eid, tuple((s.start_sec, s.end_sec) for s in spans))
            if key in seen_spans:
                report.add_warning(f"duplicate event {eid} dropped")
                continue
            seen_spans.add(key)
        except ValueError as exc:
            report.events_rejected += 1
            report.rejected_snippets.append(f"<{rec.raw_tag}>: {exc}")
            report.ok = False

    # 3. infer tracks
    tracks = _infer_tracks(events, report, duration_sec)

    report.events_recovered = len(events)
    if report.events_rejected:
        report.ok = False

    ledger = Ledger(
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        sample_id=sample_id,
        duration_sec=duration_sec,
        time_resolution_sec=TIME_RESOLUTION_SEC,  # type: ignore[arg-type]
        tracks=tracks,
        events=events,
    )
    return ledger, report


def _try_strict(text: str, sample_id: str, duration_sec: float) -> Ledger:
    from sceneledger.models.serializer import deserialize

    ledger = deserialize(text)
    # Override sample_id/duration if caller insists (strict path already
    # validated them; only override when caller-provided differs and is sane).
    if ledger.sample_id != sample_id and sample_id:
        ledger = ledger.model_copy(update={"sample_id": sample_id})
    if duration_sec and abs(ledger.duration_sec - duration_sec) > 1e-6:
        # trust the document's duration; caller's is just a default
        pass
    return ledger


def _infer_tracks(
    events: list[Event], report: ParseReport, duration_sec: float
) -> list[Track]:
    """Build minimal tracks so event.track_id references resolve.

    Track kind is inferred from the referencing event's type; spans are the
    union of referencing events' spans. Real ledgers carry richer tracks.
    """
    by_track: dict[str, list[Event]] = {}
    for e in events:
        if e.track_id is None:
            continue
        by_track.setdefault(e.track_id, []).append(e)

    tracks: list[Track] = []
    for tid in sorted(by_track):
        refs = by_track[tid]
        kind = _infer_track_kind(refs)
        spans = _union_spans([s for e in refs for s in e.spans])
        if not spans:
            continue
        tracks.append(
            Track(
                id=tid,
                kind=kind,  # type: ignore[arg-type]
                spans=spans,
                confidence=min(e.confidence for e in refs),
            )
        )
    if tracks:
        report.warnings.append("tracks inferred from events (no <tracks> header)")
        report.ok = False
    return tracks


def _infer_track_kind(refs: list[Event]) -> str:
    types = {e.type for e in refs}
    if types == {"speech"}:
        return "speech"
    if types == {"lys"}:
        return "vocal"
    if types <= {"music"}:
        return "music"
    if types <= {"sfx"}:
        return "sfx"
    return "residual"


def _union_spans(spans: list[Span]) -> list[Span]:
    """Merge overlapping/adjacent spans on the 0.1 s grid."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s.start_sec)
    merged: list[Span] = [spans[0]]
    for cur in spans[1:]:
        last = merged[-1]
        if cur.start_sec <= last.end_sec - 1e-6:
            # overlap or touch -> extend
            merged[-1] = Span(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, cur.end_sec),
            )
        else:
            merged.append(cur)
    return merged


__all__ = ["ParseReport", "RecoveredEvent", "parse_model_output"]
