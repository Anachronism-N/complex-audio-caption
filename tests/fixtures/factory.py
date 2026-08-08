"""Shared test fixtures factory.

Kept as a plain module (not a pytest fixture) so the same builders can be
imported by both unit tests and the reports-generation script.
"""

from __future__ import annotations

from sceneledger.data.schema import (
    Conditions,
    Event,
    Evidence,
    Ledger,
    Provenance,
    Span,
    Track,
)


def t(start: float, end: float, **kw) -> Span:
    return Span(start_sec=start, end_sec=end, **kw)


def ev(
    eid: str,
    etype: str,
    spans: list[Span],
    text: str,
    *,
    track_id: str | None = None,
    confidence: float = 0.9,
    verbatim: bool | None = None,
    language: str | None = None,
) -> Event:
    return Event(
        id=eid,
        type=etype,  # type: ignore[arg-type]
        track_id=track_id,
        spans=spans,
        text=text,
        verbatim=verbatim,
        language=language,
        confidence=confidence,
    )


def tr(
    tid: str,
    kind: str,
    spans: list[Span],
    *,
    identity: str | None = None,
    confidence: float = 0.9,
) -> Track:
    return Track(
        id=tid,
        kind=kind,  # type: ignore[arg-type]
        identity=identity,
        spans=spans,
        confidence=confidence,
    )


def ledger(
    sample_id: str,
    duration: float,
    tracks: list[Track] | None = None,
    events: list[Event] | None = None,
    *,
    conditions: Conditions | None = None,
    provenance: Provenance | None = None,
    language: str | None = None,
) -> Ledger:
    return Ledger(
        sample_id=sample_id,
        duration_sec=duration,
        language=language,
        conditions=conditions or Conditions(),
        tracks=tracks or [],
        events=events or [],
        provenance=provenance or Provenance(),
    )
