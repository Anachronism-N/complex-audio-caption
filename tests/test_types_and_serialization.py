from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sceneledger.serialization import parse_tagged_caption, serialize_tagged_caption
from sceneledger.types import Event, Evidence, Ledger, Span, Track


def example_ledger() -> Ledger:
    return Ledger(
        sample_id="sample",
        duration_sec=5.0,
        tracks=[
            Track(
                "T1",
                "music",
                [Span(0.0, 5.0)],
                0.9,
                evidence=Evidence(
                    method="fixture",
                    spans=[Span(0.0, 5.0)],
                    audio_support=0.95,
                    waveform_uri="stems/T1.wav",
                ),
            ),
            Track("T2", "speech", [Span(1.0, 2.2)], 0.8, identity="speaker_1"),
        ],
        events=[
            Event("E2", "speech", "T2", [Span(1.0, 2.2)], "你好 world", 0.8, True, "zh"),
            Event("E1", "music", "T1", [Span(0.0, 5.0)], "soft music", 0.9),
        ],
    )


def test_ledger_validation_and_round_trip() -> None:
    ledger = example_ledger()
    ledger.validate()
    restored = Ledger.from_dict(ledger.to_dict())
    assert restored.to_dict() == ledger.to_dict()


def test_tagged_caption_is_deterministic_and_parseable() -> None:
    caption = serialize_tagged_caption(example_ledger())
    assert caption.splitlines()[0].startswith("<music")
    restored = parse_tagged_caption(caption, sample_id="sample", duration_sec=5.0)
    assert [event.id for event in restored.events] == ["E1", "E2"]
    assert restored.events[1].text == "你好 world"
    assert restored.events[1].verbatim is True


def test_tagged_caption_escapes_xml_characters() -> None:
    ledger = example_ledger()
    ledger.events[0].text = '他说 "a < b & c"'
    caption = serialize_tagged_caption(ledger)
    restored = parse_tagged_caption(caption, sample_id="sample", duration_sec=5.0)
    assert restored.events[1].text == '他说 "a < b & c"'


def test_unknown_track_is_rejected() -> None:
    ledger = example_ledger()
    ledger.events[0].track_id = "T404"
    with pytest.raises(ValueError, match="unknown track"):
        ledger.validate()


def test_non_quantized_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        Span(0.03, 1.0).validate(2.0)


def test_json_schema_accepts_decimal_grid_values() -> None:
    schema_path = Path(__file__).parents[1] / "schemas" / "track_event_ledger.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(example_ledger().to_dict()))
    assert errors == []
