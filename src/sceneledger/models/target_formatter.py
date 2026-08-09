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

import html
import re
from dataclasses import dataclass

from sceneledger.data.schema import TIME_RESOLUTION_SEC, Event, Ledger, Span, Track

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
def format_atomic_caption(
    ledger: Ledger,
    style: str = "brief",
    cfg: StyleConfig | None = None,
    *,
    include_tracks: bool = False,
) -> str:
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
    tracks = {track.id: track for track in ledger.tracks}
    for e in events:
        text = _style_text(e.text, style, cfg)
        attrs: list[str] = []
        if include_tracks and e.track_id is not None:
            attrs.append(f'track="{html.escape(e.track_id, quote=True)}"')
            identity = tracks.get(e.track_id).identity if e.track_id in tracks else None
            if identity:
                attrs.append(f'identity="{html.escape(identity, quote=True)}"')
        opening = f"<{e.type}{' ' + ' '.join(attrs) if attrs else ''}>"
        parts: list[str] = [opening]
        for i, sp in enumerate(e.spans):
            parts.append(time_to_token(sp.start_sec))
            if i == 0:
                parts.append(text)
            parts.append(time_to_token(sp.end_sec))
        parts.append(f"</{e.type}>")
        out.append("".join(parts))
    return "".join(out)


# --------------------------------------------------------------------------- #
# atomic-token parser (so the evaluator can consume B2 output)
# --------------------------------------------------------------------------- #
_ATOMIC_EVENT_RE = re.compile(
    r"<(?P<type>speech|lys|music|sfx)(?P<attrs>\s+[^>]*)?>"
    r"(?P<body>.*?)</(?P=type)>",
    re.DOTALL | re.IGNORECASE,
)
_ATTR_RE = re.compile(r'(?P<key>[\w-]+)\s*=\s*"(?P<value>[^"]*)"')


@dataclass(frozen=True)
class ParsedAtomicEvent:
    type: str
    spans: list[tuple[float, float]]
    text: str
    track_id: str | None = None
    identity: str | None = None


def parse_atomic_events(text: str) -> list[ParsedAtomicEvent]:
    """Parse atomic events while retaining optional source attributes."""
    text = text.strip()
    if text == "<empty/>":
        return []
    results: list[ParsedAtomicEvent] = []
    for match in _ATOMIC_EVENT_RE.finditer(text):
        etype = match.group("type").lower()
        body = match.group("body")
        attrs = {
            item.group("key").lower(): html.unescape(item.group("value"))
            for item in _ATTR_RE.finditer(match.group("attrs") or "")
        }
        tokens = list(_T_TOKEN_RE.finditer(body))
        if len(tokens) < 2:
            continue
        body_text = _T_TOKEN_RE.sub("", body).strip()
        spans: list[tuple[float, float]] = []
        for index in range(0, len(tokens) - 1, 2):
            start = round(int(tokens[index].group(1)) * TIME_RESOLUTION_SEC, 6)
            end = round(int(tokens[index + 1].group(1)) * TIME_RESOLUTION_SEC, 6)
            if end > start:
                spans.append((start, end))
        if spans:
            results.append(
                ParsedAtomicEvent(
                    type=etype,
                    spans=spans,
                    text=body_text,
                    track_id=attrs.get("track") or attrs.get("track_id"),
                    identity=attrs.get("identity") or attrs.get("speaker"),
                )
            )
    return results


def is_strict_atomic_caption(text: str) -> bool:
    """Return true only when the whole output obeys the atomic grammar."""
    stripped = text.strip()
    if stripped == "<empty/>":
        return True
    matches = list(_ATOMIC_EVENT_RE.finditer(stripped))
    if not matches or "".join(match.group(0) for match in matches) != stripped:
        return False
    for match in matches:
        tokens = list(_T_TOKEN_RE.finditer(match.group("body")))
        if len(tokens) < 2 or len(tokens) % 2:
            return False
    return len(parse_atomic_events(stripped)) == len(matches)


def parse_atomic_caption(text: str) -> list[tuple[str, list[tuple[float, float]], str]]:
    """Parse an atomic-token caption into (type, spans, text) tuples.

    Tolerant: ignores surrounding prose. Returns [] for ``<empty/>`` or no tags.
    """
    return [(event.type, event.spans, event.text) for event in parse_atomic_events(text)]


def _merge_spans(spans: list[Span]) -> list[Span]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (span.start_sec, span.end_sec))
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start_sec <= previous.end_sec + 1e-6:
            merged[-1] = Span(
                start_sec=previous.start_sec,
                end_sec=max(previous.end_sec, current.end_sec),
            )
        else:
            merged.append(current)
    return merged


def atomic_to_ledger(
    text: str,
    sample_id: str,
    duration_sec: float,
    *,
    clip_to_duration: bool = False,
) -> Ledger:
    """Parse an atomic-token caption back into a source-aware Ledger.

    Legacy targets without ``track`` attributes remain supported and are
    grouped by event type. Source-aware targets preserve multiple concurrent
    speech or vocal tracks.
    """
    parsed = parse_atomic_events(text)
    events: list[Event] = []
    track_kind: dict[str, str] = {}
    track_identity: dict[str, str | None] = {}
    track_spans: dict[str, list[Span]] = {}
    track_id_by_type: dict[str, str] = {}
    for parsed_event in parsed:
        etype = parsed_event.type
        expected_kind = etype if etype != "lys" else "vocal"
        explicit_tid = parsed_event.track_id
        if explicit_tid is not None and not re.fullmatch(r"T[0-9]+", explicit_tid):
            explicit_tid = None
        if explicit_tid is not None and explicit_tid in track_kind:
            if track_kind[explicit_tid] != expected_kind:
                explicit_tid = None
        if explicit_tid is None:
            if etype not in track_id_by_type:
                used = set(track_kind) | set(track_id_by_type.values())
                next_index = 1
                while f"T{next_index}" in used:
                    next_index += 1
                track_id_by_type[etype] = f"T{next_index}"
            tid = track_id_by_type[etype]
        else:
            tid = explicit_tid

        span_objects: list[Span] = []
        for start, end in parsed_event.spans:
            if clip_to_duration:
                start = min(max(0.0, start), duration_sec)
                end = min(max(0.0, end), duration_sec)
            if end > start:
                span_objects.append(Span(start_sec=start, end_sec=end))
        if not span_objects:
            continue
        track_kind.setdefault(tid, expected_kind)
        if parsed_event.identity:
            track_identity.setdefault(tid, parsed_event.identity)
        track_spans.setdefault(tid, []).extend(span_objects)
        events.append(
            Event(
                id=f"E{len(events) + 1:03d}",
                type=etype,  # type: ignore[arg-type]
                track_id=tid,
                spans=span_objects,
                text=parsed_event.text or "undescribed event",
                verbatim=True if etype == "lys" else None,
                confidence=0.9,
            )
        )
    tracks = [
        Track(
            id=tid,
            kind=track_kind[tid],  # type: ignore[arg-type]
            identity=track_identity.get(tid),
            spans=_merge_spans(track_spans[tid]),
            confidence=0.9,
        )
        for tid in sorted(track_kind, key=lambda value: int(value[1:]))
    ]
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
    include_tracks: bool = False,
    output_mode: str | None = None,
) -> str:
    if output_mode not in {None, "atomic", "xml"}:
        raise ValueError("output_mode must be one of: atomic, xml, or None")
    lyrics_line = (
        "Return sung lyrics as <lys> events; if lyrics are not clearly "
        'intelligible, describe them inside <music> as "unclear vocals" '
        "rather than guessing words.\n"
        if include_lyrics
        else ""
    )
    grammar = ""
    if output_mode == "atomic":
        source_attrs = (
            ' Add track="Tn" and optional identity="Sn" attributes to every '
            "event tag; reuse a track ID for the same audible source."
            if include_tracks
            else ""
        )
        grammar = (
            "Return only concatenated tagged events, with no prose or Markdown.\n"
            "Atomic grammar: <TYPE><|t_SSS|>event text<|t_EEE|></TYPE>, where "
            "TYPE is speech, music, sfx"
            + (", or lys" if include_lyrics else "")
            + " and SSS/EEE are decisecond indices from 000 to 300.\n"
            + source_attrs
            + "\nExample: "
            + (
                '<speech track="T1" identity="S1"><|t_007|>a person says hello'
                "<|t_029|></speech>\n"
                if include_tracks
                else "<speech><|t_007|>a person says hello<|t_029|></speech>\n"
            )
        )
    elif output_mode == "xml":
        grammar = (
            "Return only one tagged event per line, with no prose or Markdown.\n"
            'XML grammar: <TYPE t="START-END">event text</TYPE>; use comma-separated '
            "START-END pairs for repeated spans.\n"
        )
    return (
        "Describe every audible event in the audio.\n"
        "Return speech, sung lyrics, music, and sound effects as separate typed events.\n"
        "Events may overlap and may contain multiple time spans.\n"
        f"[style={style}, merge={merge_s}s, activity={activity}, resolution={resolution_s}s]\n"
        f"{lyrics_line}"
        f"{grammar}"
        "Do not infer events that are not acoustically supported."
    ).strip()


__all__ = [
    "MAX_TIMESTAMP_SEC",
    "ParsedAtomicEvent",
    "StyleConfig",
    "T_TOKEN_COUNT",
    "T_TOKEN_FIRST",
    "T_TOKEN_LAST",
    "atomic_to_ledger",
    "canonical_prompt",
    "format_atomic_caption",
    "format_xml_caption",
    "is_strict_atomic_caption",
    "parse_atomic_caption",
    "parse_atomic_events",
    "time_to_token",
    "token_to_time",
]
