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

from sceneledger.data.schema import TIME_RESOLUTION_SEC, Event, Ledger

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
# atomic-token parser (so the evaluator can consume B2 output)
# --------------------------------------------------------------------------- #
_TAG_OPEN_RE = re.compile(r"<(speech|lys|music|sfx)>")
_TAG_CLOSE_RE = re.compile(r"</(speech|lys|music|sfx)>")


def parse_atomic_caption(text: str) -> list[tuple[str, list[tuple[float, float]], str]]:
    """Parse an atomic-token caption into (type, spans, text) tuples.

    Tolerant: ignores surrounding prose. Returns [] for ``<empty/>`` or no tags.
    """
    text = text.strip()
    if text == "<empty/>":
        return []
    results: list[tuple[str, list[tuple[float, float]], str]] = []
    pos = 0
    while pos < len(text):
        m_open = _TAG_OPEN_RE.search(text, pos)
        if not m_open:
            break
        etype = m_open.group(1)
        start = m_open.end()
        m_close = _TAG_CLOSE_RE.search(text, start)
        if not m_close:
            break
        body = text[start:m_close.start()]
        # extract token positions and text
        tokens = list(_T_TOKEN_RE.finditer(body))
        if len(tokens) < 2:
            pos = m_close.end()
            continue
        # text = body with tokens stripped
        body_text = _T_TOKEN_RE.sub("", body).strip()
        # pair tokens into (start, end) spans
        spans: list[tuple[float, float]] = []
        i = 0
        while i + 1 < len(tokens):
            s = round(int(tokens[i].group(1)) * TIME_RESOLUTION_SEC, 6)
            e = round(int(tokens[i + 1].group(1)) * TIME_RESOLUTION_SEC, 6)
            if e > s:
                spans.append((s, e))
            i += 2
        if spans:
            results.append((etype, spans, body_text))
        pos = m_close.end()
    return results


def atomic_to_ledger(
    text: str, sample_id: str, duration_sec: float
) -> Ledger:
    """Parse an atomic-token caption back into a :class:`Ledger`."""
    from sceneledger.data.schema import Span

    parsed = parse_atomic_caption(text)
    events: list[Event] = []
    tracks: list = []
    track_id_by_type: dict[str, str] = {}
    for i, (etype, spans, ev_text) in enumerate(parsed):
        if etype not in track_id_by_type:
            tid = f"T{len(track_id_by_type) + 1}"
            track_id_by_type[etype] = tid
            kind = etype if etype != "lys" else "vocal"
            tracks.append(
                __import__(
                    "sceneledger.data.schema", fromlist=["Track"]
                ).Track(
                    id=tid,
                    kind=kind,  # type: ignore[arg-type]
                    spans=[Span(start_sec=s, end_sec=e) for s, e in spans],
                    confidence=0.9,
                )
            )
        tid = track_id_by_type[etype]
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
        grammar = (
            "Return only concatenated tagged events, with no prose or Markdown.\n"
            "Atomic grammar: <TYPE><|t_SSS|>event text<|t_EEE|></TYPE>, where "
            "TYPE is speech, music, sfx"
            + (", or lys" if include_lyrics else "")
            + " and SSS/EEE are decisecond indices from 000 to 300.\n"
            "Example: <speech><|t_007|>a person says hello<|t_029|></speech>\n"
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
    "StyleConfig",
    "T_TOKEN_COUNT",
    "T_TOKEN_FIRST",
    "T_TOKEN_LAST",
    "atomic_to_ledger",
    "canonical_prompt",
    "format_atomic_caption",
    "format_xml_caption",
    "parse_atomic_caption",
    "time_to_token",
    "token_to_time",
]
