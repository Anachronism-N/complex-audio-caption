from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .serialization import serialize_tagged_caption
from .types import Event, Ledger, Span, quantize_time

NEGATIVE_TYPES = (
    "hallucination_insert",
    "event_omission",
    "timestamp_shift",
    "event_type_swap",
    "track_pointer_swap",
    "event_duplication",
    "overlong_span",
)


def build_preference_rows(
    ledgers: Iterable[Ledger],
    audio_paths: Mapping[str, str | Path],
    *,
    negatives_per_sample: int = 4,
    seed: int = 20260808,
    prompt: str = (
        "Transcribe and describe every audible speech, lyric, music and sound event "
        "with 0.1-second timestamps."
    ),
) -> list[dict]:
    """Create deterministic DPO/RLHF rows with structure-aware hard negatives."""
    if negatives_per_sample < 1:
        raise ValueError("negatives_per_sample must be positive")
    rows: list[dict] = []
    for sample_index, ledger in enumerate(ledgers):
        if ledger.sample_id not in audio_paths:
            raise ValueError(f"No audio path for {ledger.sample_id}")
        chosen = serialize_tagged_caption(ledger)
        rng = np.random.default_rng(seed + sample_index)
        order = list(rng.permutation(NEGATIVE_TYPES))
        for negative_index in range(negatives_per_sample):
            requested = order[negative_index % len(order)]
            rejected_ledger, applied = make_hard_negative(ledger, requested, rng)
            rejected = serialize_tagged_caption(rejected_ledger)
            if rejected == chosen:
                raise AssertionError(f"Negative operation {applied} made no caption change")
            rows.append(
                {
                    "sample_id": ledger.sample_id,
                    "preference_id": f"{ledger.sample_id}:neg{negative_index:02d}",
                    "audio_path": str(Path(audio_paths[ledger.sample_id]).resolve()),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "negative_type": applied,
                    "seed": seed + sample_index,
                }
            )
    return rows


def write_preference_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def make_hard_negative(
    ledger: Ledger, operation: str, rng: np.random.Generator
) -> tuple[Ledger, str]:
    if operation not in NEGATIVE_TYPES:
        raise ValueError(f"Unknown negative operation: {operation}")
    value = copy.deepcopy(ledger.to_dict())
    negative = Ledger.from_dict(value)
    if operation == "hallucination_insert" or not negative.events:
        _insert_hallucination(negative, rng)
        applied = "hallucination_insert"
    elif operation == "event_omission":
        removed = negative.events.pop(int(rng.integers(len(negative.events))))
        for event in negative.events:
            event.relations = [
                relation
                for relation in event.relations
                if relation.get("target_event_id") != removed.id
            ]
        applied = operation
    elif operation == "timestamp_shift":
        event = negative.events[int(rng.integers(len(negative.events)))]
        if not _shift_event(event, negative.duration_sec):
            _swap_type(event)
            applied = "event_type_swap"
        else:
            applied = operation
    elif operation == "event_type_swap":
        _swap_type(negative.events[int(rng.integers(len(negative.events)))])
        applied = operation
    elif operation == "track_pointer_swap":
        if len(negative.tracks) < 2:
            _swap_type(negative.events[int(rng.integers(len(negative.events)))])
            applied = "event_type_swap"
        else:
            event = negative.events[int(rng.integers(len(negative.events)))]
            choices = [track.id for track in negative.tracks if track.id != event.track_id]
            event.track_id = choices[int(rng.integers(len(choices)))]
            applied = operation
    elif operation == "event_duplication":
        event = copy.deepcopy(negative.events[int(rng.integers(len(negative.events)))])
        event.id = _next_event_id(negative)
        event.text = f"again, {event.text}"
        negative.events.append(event)
        applied = operation
    elif operation == "overlong_span":
        event = negative.events[int(rng.integers(len(negative.events)))]
        event.spans = [Span(0.0, _grid_floor(negative.duration_sec))]
        applied = operation
    else:  # pragma: no cover
        raise AssertionError(operation)
    negative.events.sort(key=lambda event: (event.onset, event.id))
    negative.validate()
    return negative, applied


def _insert_hallucination(ledger: Ledger, rng: np.random.Generator) -> None:
    event_types = ["speech", "lys", "music", "sfx"]
    event_type = event_types[int(rng.integers(len(event_types)))]
    duration = min(1.0, ledger.duration_sec)
    maximum_start = max(0.0, ledger.duration_sec - duration)
    start = quantize_time(float(rng.uniform(0.0, maximum_start))) if maximum_start else 0.0
    end = min(quantize_time(start + duration), quantize_time(ledger.duration_sec))
    if end <= start:
        end = quantize_time(min(ledger.duration_sec, start + 0.1))
    descriptions = {
        "speech": "an unsupported speaker says something",
        "lys": "unsupported lyrics are sung",
        "music": "an unsupported melody plays",
        "sfx": "an unsupported bell rings",
    }
    ledger.events.append(
        Event(
            id=_next_event_id(ledger),
            type=event_type,  # type: ignore[arg-type]
            track_id=ledger.tracks[0].id if ledger.tracks else None,
            spans=[Span(start, end)],
            text=descriptions[event_type],
            confidence=1.0,
        )
    )


def _shift_event(event: Event, duration_sec: float) -> bool:
    if max(span.end_sec for span in event.spans) + 0.5 <= duration_sec + 1e-6:
        delta = 0.5
    elif min(span.start_sec for span in event.spans) >= 0.5:
        delta = -0.5
    else:
        return False
    event.spans = [
        Span(quantize_time(span.start_sec + delta), quantize_time(span.end_sec + delta))
        for span in event.spans
    ]
    return True


def _swap_type(event: Event) -> None:
    choices = [item for item in ("speech", "lys", "music", "sfx") if item != event.type]
    event.type = choices[0]  # type: ignore[assignment]
    event.verbatim = None if event.type not in {"speech", "lys"} else False


def _next_event_id(ledger: Ledger) -> str:
    existing = {
        int(event.id[1:])
        for event in ledger.events
        if event.id.startswith("E") and event.id[1:].isdigit()
    }
    candidate = 1
    while candidate in existing:
        candidate += 1
    return f"E{candidate}"


def _grid_floor(value: float, resolution_sec: float = 0.1) -> float:
    return round(math.floor((value + 1e-8) / resolution_sec) * resolution_sec, 6)
