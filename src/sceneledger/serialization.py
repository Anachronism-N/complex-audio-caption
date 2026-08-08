from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.sax.saxutils import escape, quoteattr

from .types import Event, Ledger, Span, Track, quantize_time

_TYPE_ORDER = {"music": 0, "speech": 1, "lys": 2, "sfx": 3}
_TAG_TO_TRACK_KIND = {"speech": "speech", "lys": "vocal", "music": "music", "sfx": "sfx"}


def serialize_tagged_caption(ledger: Ledger) -> str:
    """Serialize events deterministically; track metadata remains in the canonical JSON ledger."""
    ledger.validate()
    lines: list[str] = []
    events = sorted(
        ledger.events, key=lambda event: (event.onset, _TYPE_ORDER[event.type], event.id)
    )
    for event in events:
        attrs = {
            "id": event.id,
            "track": event.track_id,
            "t": _format_spans(event.spans),
            "confidence": f"{event.confidence:.3f}",
            "language": event.language,
            "verbatim": None if event.verbatim is None else str(event.verbatim).lower(),
        }
        rendered_attrs = " ".join(
            f"{name}={quoteattr(value)}" for name, value in attrs.items() if value is not None
        )
        lines.append(f"<{event.type} {rendered_attrs}>{escape(event.text)}</{event.type}>")
    return "\n".join(lines)


def parse_tagged_caption(
    text: str,
    *,
    sample_id: str,
    duration_sec: float,
    strict: bool = True,
) -> Ledger:
    """Parse the constrained readable format back into a canonical ledger.

    In non-strict mode, surrounding prose is discarded. Tag bodies are never repaired or guessed.
    """
    if not strict:
        matches = re.findall(r"<(speech|lys|music|sfx)\b[^>]*>.*?</\1>", text, re.DOTALL)
        if not matches:
            raise ValueError("No supported caption tags found")
        fragments = re.finditer(
            r"<(?:speech|lys|music|sfx)\b[^>]*>.*?</(?:speech|lys|music|sfx)>",
            text,
            re.DOTALL,
        )
        text = "\n".join(match.group(0) for match in fragments)

    try:
        root = ET.fromstring(f"<root>{text}</root>")
    except ET.ParseError as exc:
        raise ValueError(f"Invalid tagged caption: {exc}") from exc

    events: list[Event] = []
    track_spans: dict[str, list[Span]] = defaultdict(list)
    track_kinds: dict[str, str] = {}
    for index, element in enumerate(root, 1):
        if element.tag not in _TAG_TO_TRACK_KIND:
            raise ValueError(f"Unsupported tag: {element.tag}")
        event_id = element.attrib.get("id", f"E{index}")
        track_id = element.attrib.get("track")
        if track_id is None:
            track_id = f"T{index}"
        spans = _parse_spans(element.attrib.get("t", ""), duration_sec)
        confidence = float(element.attrib.get("confidence", 1.0))
        verbatim_text = element.attrib.get("verbatim")
        verbatim = None if verbatim_text is None else verbatim_text.lower() == "true"
        event = Event(
            id=event_id,
            type=element.tag,
            track_id=track_id,
            spans=spans,
            text="".join(element.itertext()).strip(),
            confidence=confidence,
            language=element.attrib.get("language"),
            verbatim=verbatim,
        )
        events.append(event)
        track_spans[track_id].extend(spans)
        track_kinds[track_id] = _TAG_TO_TRACK_KIND[element.tag]

    tracks = [
        Track(
            id=track_id,
            kind=track_kinds[track_id],  # type: ignore[arg-type]
            spans=_merge_spans(spans),
            confidence=min(event.confidence for event in events if event.track_id == track_id),
        )
        for track_id, spans in track_spans.items()
    ]
    ledger = Ledger(sample_id=sample_id, duration_sec=duration_sec, tracks=tracks, events=events)
    ledger.validate()
    return ledger


def _format_spans(spans: list[Span]) -> str:
    return ",".join(f"{span.start_sec:.1f}-{span.end_sec:.1f}" for span in spans)


def _parse_spans(value: str, duration_sec: float) -> list[Span]:
    if not value:
        raise ValueError("Every event must have a t attribute")
    spans: list[Span] = []
    for item in value.split(","):
        parts = item.strip().split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid time span: {item!r}")
        start, end = (quantize_time(float(part)) for part in parts)
        spans.append(Span(start, end))
    merged = _merge_spans(spans)
    for span in merged:
        span.validate(duration_sec)
    return merged


def _merge_spans(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in sorted(spans):
        if merged and span.start_sec <= merged[-1].end_sec + 1e-6:
            previous = merged[-1]
            merged[-1] = Span(previous.start_sec, max(previous.end_sec, span.end_sec))
        else:
            merged.append(span)
    return merged
