"""Unit tests for the TAC target formatter."""

from __future__ import annotations

from fixtures.factory import ev, ledger, t, tr

from sceneledger.data.schema import Ledger
from sceneledger.models.target_formatter import (
    T_TOKEN_COUNT,
    atomic_to_ledger,
    canonical_prompt,
    format_atomic_caption,
    format_xml_caption,
    parse_atomic_caption,
    time_to_token,
    token_to_time,
)


def _sample_ledger() -> Ledger:
    return ledger(
        "ex01",
        12.8,
        tracks=[
            tr("T1", "music", [t(0.0, 12.8)]),
            tr("T2", "speech", [t(0.7, 2.9)]),
            tr("T3", "sfx", [t(4.6, 4.9), t(6.0, 6.3)]),
        ],
        events=[
            ev("E1", "music", [t(0.0, 12.8)], text="light electronic beat", track_id="T1", confidence=0.96),
            ev("E2", "speech", [t(0.7, 2.9)], text="a man says hello world quickly", track_id="T2", confidence=0.94),
            ev("E3", "sfx", [t(4.6, 4.9), t(6.0, 6.3)], text="glass break twice", track_id="T3", confidence=0.91),
        ],
    )


def test_time_to_token_format():
    assert time_to_token(0.0) == "<|t_000|>"
    assert time_to_token(0.7) == "<|t_007|>"
    assert time_to_token(12.8) == "<|t_128|>"
    assert time_to_token(30.0) == "<|t_300|>"
    assert time_to_token(31.0) == "<|t_300|>"  # clamped
    assert token_to_time("<|t_046|>") == 4.6


def test_xml_caption_format_and_order():
    cap = format_xml_caption(_sample_ledger(), style="detailed")
    # onset order: music(0.0) < speech(0.7) < sfx(4.6)
    assert cap.index("<music") < cap.index("<speech") < cap.index("<sfx")
    assert 't="0-12.8"' in cap
    assert 't="4.6-4.9,6-6.3"' in cap  # multi-span


def test_atomic_caption_format():
    cap = format_atomic_caption(_sample_ledger(), style="detailed")
    # music first (onset 0.0)
    assert cap.startswith("<music><|t_000|>")
    # speech event with start/end tokens
    assert "<speech><|t_007|>" in cap
    assert "<|t_029|></speech>" in cap
    # multi-span sfx: tokens 046-049 then 060-063
    assert "<|t_046|>" in cap and "<|t_049|>" in cap
    assert "<|t_060|>" in cap and "<|t_063|>" in cap


def test_atomic_round_trip():
    lg = _sample_ledger()
    cap = format_atomic_caption(lg, style="detailed")
    parsed = parse_atomic_caption(cap)
    assert len(parsed) == 3
    # types preserved
    types = {p[0] for p in parsed}
    assert types == {"music", "speech", "sfx"}
    # sfx multi-span preserved
    sfx = next(p for p in parsed if p[0] == "sfx")
    assert len(sfx[1]) == 2
    assert sfx[1][0] == (4.6, 4.9)
    assert sfx[1][1] == (6.0, 6.3)


def test_atomic_to_ledger_schema_valid():
    lg = _sample_ledger()
    cap = format_atomic_caption(lg, style="detailed")
    back = atomic_to_ledger(cap, "ex01", 12.8)
    Ledger.model_validate(back.model_dump())
    assert len(back.events) == 3


def test_empty_scene_emits_empty_tag():
    lg = ledger("empty", 5.0)
    assert format_xml_caption(lg) == "<empty/>"
    assert format_atomic_caption(lg) == "<empty/>"
    assert parse_atomic_caption("<empty/>") == []


def test_style_keyword_trims_text():
    lg = ledger("s", 5.0, events=[ev("E1", "speech", [t(0, 1)], text="one two three four five six seven")])
    kw = format_xml_caption(lg, style="keyword")
    # keyword limits to 4 tokens
    assert "five" not in kw or "…" in kw


def test_canonical_prompt_includes_lyrics_toggle():
    no_ly = canonical_prompt(include_lyrics=False)
    with_ly = canonical_prompt(include_lyrics=True)
    assert "<lys>" not in no_ly
    assert "<lys>" in with_ly
    assert "style=brief" in no_ly


def test_canonical_prompt_can_specify_atomic_grammar():
    prompt = canonical_prompt(output_mode="atomic")
    assert "Atomic grammar" in prompt
    assert "<|t_SSS|>" in prompt
    assert "no prose or Markdown" in prompt


def test_token_count_matches_config():
    # experiment_matrix.yaml: count 301 (0.0..30.0 inclusive at 0.1s)
    assert T_TOKEN_COUNT == 301
