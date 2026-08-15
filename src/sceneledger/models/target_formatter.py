"""Format a :class:`Ledger` into TAC-style training targets.

Two serialization modes are needed by the baselines (``docs/06`` §5):

* ``xml``      -- B1 static SFT: the human-readable tagged caption with
  ``t="start-end"`` attributes (what :mod:`serializer` already produces).
  Ordinary token CE is used.
* ``atomic``   -- B2 TAC paper-spec: timestamps are emitted as atomic tokens
  ``<|t_000|>``..``<|t_300|>`` (one per 0.1 s, 0.0–30.0 s, 301 tokens per
  ``configs/experiment_matrix.yaml``). Time-weighted CE upweights these.

Three ``style`` values control how verbose the event text is:

* ``keyword``  -- a single phrase per event (e.g. "glass break").
* ``brief``    -- one short sentence.
* ``detailed`` -- the full caption text.

The same formatter drives both teacher targets and (for the mock B0) model
output, so the parser/evaluator can round-trip either mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sceneledger.data.schema import TIME_RESOLUTION_SEC, Event, Ledger, Track

# Atomic timestamp token vocabulary (configs/experiment_matrix.yaml).
T_TOKEN_FIRST = "<|t_000|>"
T_TOKEN_LAST = "<|t_300|>"
T_TOKEN_COUNT = 301
MAX_TIMESTAMP_SEC = 30.0

_TYPE_ORDER = {"speech": 0, "lys": 1, "music": 2, "sfx": 3}
_T_TOKEN_RE = re.compile(r"<\|t_(\d{3})\|>")


def time_to_token(t_sec: float) -> str:
    """``0.7`` -> ``<|t_007|>``. Clamped to [0, 300]."""
    idx = int(round(t_sec / TIME_RESOLUTION_SEC))
    idx = max(0, min(T_TOKEN_COUNT - 1, idx))
    return f"<|t_{idx:03d}|>"


def token_to_time(token: str) -> float | None:
    m = _T_TOKEN_RE.fullmatch(token.strip())
    if not m:
        return None
    return round(int(m.group(1)) * TIME_RESOLUTION_SEC, 6)


# --------------------------------------------------------------------------- #
# style-based text trimming
# --------------------------------------------------------------------------- #
@dataclass
class StyleConfig:
    keyword_max_tokens: int = 4
    brief_max_tokens: int = 12
    detailed_max_tokens: int = 60


def _style_text(text: str, style: str, cfg: StyleConfig) -> str:
    words = text.split()
    if style == "keyword":
        limit = cfg.keyword_max_tokens
    elif style == "brief":
        limit = cfg.brief_max_tokens
    else:  # detailed
        limit = cfg.detailed_max_tokens
    if len(words) <= limit:
        return text
    trimmed = " ".join(words[:limit])
    return trimmed + "…"


def _ordered_events(ledger: Ledger) -> list[Event]:
    """Stable order: onset asc, then type order, then id (docs/06 §5.2)."""
    return sorted(
        ledger.events,
        key=lambda e: (
            round(e.start_sec(), 6),
            _TYPE_ORDER.get(e.type, 9),
            e.id,
        ),
    )


# --------------------------------------------------------------------------- #
# XML (B1) target
# --------------------------------------------------------------------------- #
def format_xml_caption(ledger: Ledger, style: str = "brief", cfg: StyleConfig | None = None) -> str:
    """B1 target: tagged caption with ``t`` attributes, onset-ordered.

    Empty scene -> ``<empty/>``.
    """
    cfg = cfg or StyleConfig()
    events = _ordered_events(ledger)
    if not events:
        return "<empty/>"
    lines: list[str] = []
    for e in events:
        spans_str = ",".join(f"{s.start_sec:g}-{s.end_sec:g}" for s in e.spans)
        attrs = [f't="{spans_str}"', f'confidence="{e.confidence:g}"']
        if e.track_id is not None:
            attrs.append(f'track="{e.track_id}"')
        text = _style_text(e.text, style, cfg)
        lines.append(f"<{e.type} {' '.join(attrs)}>{text}</{e.type}>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# atomic-token (B2) target
# --------------------------------------------------------------------------- #
def format_atomic_caption(ledger: Ledger, style: str = "brief", cfg: StyleConfig | None = None) -> str:
    """B2 target: atomic timestamp tokens interleaved with text.

    Each event: ``<type><|t_s1|>text<|t_e1|><|t_s2|>text2<|t_e2|></type>``.
    Multi-span events repeat the (start, end, text) pattern per span; since the
    source caption is shared across spans, only the first span carries the full
    text and subsequent spans emit a bare ``<|t_s|><|t_e|>`` pair.
    """
    cfg = cfg or StyleConfig()
    events = _ordered_events(ledger)
    if not events:
        return "<empty/>"
    out: list[str] = []
    for e in events:
        text = _style_text(e.text, style, cfg)
        parts: list[str] = [f"<{e.type}>"]
        for i, sp in enumerate(e.spans):
            parts.append(time_to_token(sp.start_sec))
            if i == 0:
                parts.append(text)
            parts.append(time_to_token(sp.end_sec))
        parts.append(f"</{e.type}>")
        out.append("".join(parts))
    return "".join(out)


# --------------------------------------------------------------------------- #
# slot-aware target (S1-inspired autoregressive)
# --------------------------------------------------------------------------- #
def format_slot_aware_caption(
    ledger: Ledger, style: str = "brief", cfg: StyleConfig | None = None
) -> str:
    """Track-aware target: event count prefix + slot-wrapped events.

    Format: ``<n>N</n><slot><type track="T1">...</type></slot>...``

    The count prefix teaches the model to predict how many events to generate
    (reducing hallucination/omission). The slot wrappers give explicit event
    boundary structure.  A stable, reusable ``track`` reference is required so
    two events from one source can be distinguished from two independent
    sources of the same type. Combined with shuffle_events in training, this
    is a lightweight autoregressive approximation of S1 set prediction.
    """
    cfg = cfg or StyleConfig()
    events = _ordered_events(ledger)
    if not events:
        return "<n>0</n><empty/>"
    n = len(events)
    out: list[str] = [f"<n>{n}</n>"]
    for e in events:
        text = _style_text(e.text, style, cfg)
        track_attr = f' track="{e.track_id}"' if e.track_id is not None else ""
        parts: list[str] = [f"<{e.type}{track_attr}>"]
        for i, sp in enumerate(e.spans):
            parts.append(time_to_token(sp.start_sec))
            if i == 0:
                parts.append(text)
            parts.append(time_to_token(sp.end_sec))
        parts.append(f"</{e.type}>")
        out.append(f"<slot>{''.join(parts)}</slot>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# atomic-token parser (so the evaluator can consume B2 output)
# --------------------------------------------------------------------------- #
_TAG_OPEN_RE = re.compile(
    r"<(speech|lys|music|sfx)(?P<attrs>\s+[^>]*)?>"
)
_TAG_CLOSE_RE = re.compile(r"</(speech|lys|music|sfx)>")
_TRACK_ATTR_RE = re.compile(r'\btrack\s*=\s*"(T[0-9]+)"')


def _parse_atomic_records(
    text: str,
) -> list[tuple[str, list[tuple[float, float]], str, str | None]]:
    """Parse atomic events while retaining explicit track references."""
    text = text.strip()
    if text == "<empty/>":
        return []
    results: list[tuple[str, list[tuple[float, float]], str, str | None]] = []
    pos = 0
    while pos < len(text):
        m_open = _TAG_OPEN_RE.search(text, pos)
        if not m_open:
            break
        etype = m_open.group(1)
        attrs = m_open.group("attrs") or ""
        track_match = _TRACK_ATTR_RE.search(attrs)
        track_id = track_match.group(1) if track_match is not None else None
        start = m_open.end()
        m_close = _TAG_CLOSE_RE.search(text, start)
        if not m_close or m_close.group(1) != etype:
            pos = start
            continue
        body = text[start:m_close.start()]
        tokens = list(_T_TOKEN_RE.finditer(body))
        if len(tokens) < 2:
            pos = m_close.end()
            continue
        body_text = _T_TOKEN_RE.sub("", body).strip()
        spans: list[tuple[float, float]] = []
        for index in range(0, len(tokens) - 1, 2):
            start_sec = round(
                int(tokens[index].group(1)) * TIME_RESOLUTION_SEC, 6
            )
            end_sec = round(
                int(tokens[index + 1].group(1)) * TIME_RESOLUTION_SEC, 6
            )
            if end_sec > start_sec:
                spans.append((start_sec, end_sec))
        if spans:
            results.append((etype, spans, body_text, track_id))
        pos = m_close.end()
    return results


def parse_atomic_caption(text: str) -> list[tuple[str, list[tuple[float, float]], str]]:
    """Parse an atomic-token caption into (type, spans, text) tuples.

    Tolerant: ignores surrounding prose. Returns [] for ``<empty/>`` or no tags.
    """
    return [
        (etype, spans, body_text)
        for etype, spans, body_text, _track_id in _parse_atomic_records(text)
    ]


def atomic_track_ids_complete(text: str) -> bool:
    """Whether every recoverable atomic event carries an explicit track ID."""
    records = _parse_atomic_records(text)
    return bool(records) and all(track_id is not None for *_, track_id in records)


def atomic_to_ledger(
    text: str, sample_id: str, duration_sec: float
) -> Ledger:
    """Parse an atomic-token caption back into a :class:`Ledger`."""
    from sceneledger.data.schema import Span

    parsed = _parse_atomic_records(text)
    events: list[Event] = []
    tracks: list = []
    track_id_by_type: dict[str, str] = {}
    used_track_ids = {
        track_id for _etype, _spans, _text, track_id in parsed if track_id
    }

    def _fallback_track_id(etype: str) -> str:
        existing = track_id_by_type.get(etype)
        if existing is not None:
            return existing
        index = 1
        while f"T{index}" in used_track_ids:
            index += 1
        track_id = f"T{index}"
        used_track_ids.add(track_id)
        track_id_by_type[etype] = track_id
        return track_id

    for i, (etype, spans, ev_text, explicit_track_id) in enumerate(parsed):
        tid = explicit_track_id or _fallback_track_id(etype)
        events.append(
            Event(
                id=f"E{i + 1:03d}",
                type=etype,  # type: ignore[arg-type]
                track_id=tid,
                spans=[Span(start_sec=s, end_sec=e) for s, e in spans],
                text=ev_text or "undescribed event",
                confidence=0.9,
            )
        )

    by_track: dict[str, list[Event]] = {}
    for event in events:
        if event.track_id is not None:
            by_track.setdefault(event.track_id, []).append(event)
    for track_id in sorted(by_track, key=lambda value: int(value[1:])):
        referenced = by_track[track_id]
        event_types = {event.type for event in referenced}
        if event_types == {"speech"}:
            kind = "speech"
        elif event_types == {"lys"}:
            kind = "vocal"
        elif event_types == {"music"}:
            kind = "music"
        elif event_types == {"sfx"}:
            kind = "sfx"
        else:
            kind = "residual"
        ordered_spans = sorted(
            [span for event in referenced for span in event.spans],
            key=lambda span: (span.start_sec, span.end_sec),
        )
        merged_spans: list[Span] = []
        for span in ordered_spans:
            if merged_spans and span.start_sec <= merged_spans[-1].end_sec:
                merged_spans[-1] = Span(
                    start_sec=merged_spans[-1].start_sec,
                    end_sec=max(merged_spans[-1].end_sec, span.end_sec),
                )
            else:
                merged_spans.append(span)
        tracks.append(
            Track(
                id=track_id,
                kind=kind,
                spans=merged_spans,
                confidence=min(event.confidence for event in referenced),
            )
        )
    return Ledger(
        sample_id=sample_id,
        duration_sec=duration_sec,
        tracks=tracks,
        events=events,
    )


# --------------------------------------------------------------------------- #
# canonical prompt (docs/06 §5.1)
# --------------------------------------------------------------------------- #
def canonical_prompt(
    style: str = "brief",
    merge_s: float = 0.25,
    activity: float = 0.05,
    resolution_s: float = 0.1,
    include_lyrics: bool = False,
    track_aware: bool = False,
) -> str:
    lyrics_line = (
        "Return sung lyrics as <lys> events; if lyrics are not clearly "
        'intelligible, describe them inside <music> as "unclear vocals" '
        "rather than guessing words.\n"
        if include_lyrics
        else ""
    )
    track_line = (
        'Give every event a track="T1", track="T2", ... attribute. Reuse the '
        "same track ID only for events produced by the same persistent source; "
        "different speakers or sound sources must use different track IDs.\n"
        if track_aware
        else ""
    )
    return (
        "Describe every audible event in the audio.\n"
        "Return speech, sung lyrics, music, and sound effects as separate typed events.\n"
        "Events may overlap and may contain multiple time spans.\n"
        f"[style={style}, merge={merge_s}s, activity={activity}, resolution={resolution_s}s]\n"
        f"{lyrics_line}"
        f"{track_line}"
        "Do not infer events that are not acoustically supported."
    ).strip()


__all__ = [
    "MAX_TIMESTAMP_SEC",
    "StyleConfig",
    "T_TOKEN_COUNT",
    "T_TOKEN_FIRST",
    "T_TOKEN_LAST",
    "atomic_to_ledger",
    "atomic_track_ids_complete",
    "canonical_prompt",
    "format_atomic_caption",
    "format_slot_aware_caption",
    "format_xml_caption",
    "parse_atomic_caption",
    "time_to_token",
    "token_to_time",
]
