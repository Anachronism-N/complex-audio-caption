from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..data.audio import activity_to_spans
from ..types import Event, Evidence, Ledger, Span, Track

TRACK_KINDS = ("speech", "vocal", "music", "sfx", "ambience", "residual")
EVENT_TYPES = ("speech", "lys", "music", "sfx")


def decode_slot_arrays(
    outputs: Mapping[str, Any],
    *,
    sample_id: str,
    duration_sec: float,
    event_texts: Sequence[str] | Mapping[int, str] | None = None,
    frame_sec: float = 0.1,
    track_threshold: float = 0.5,
    event_threshold: float = 0.5,
    activity_threshold: float = 0.5,
) -> Ledger:
    """Decode one TrackEventSlot output batch into a validated SceneLedger.

    Inputs may be NumPy arrays, CPU tensors, or batch-size-one arrays. Language
    generation is deliberately decoupled: pass text decoded from each local event
    feature through ``event_texts``.
    """
    track_presence = _one_sample(outputs["track_presence_logits"], 1)
    track_types = _one_sample(outputs["track_type_logits"], 2)
    track_activity = _one_sample(outputs["track_activity_logits"], 2)
    track_audibility = _one_sample(outputs.get("track_audibility_logits", track_presence), 1)
    eventness = _one_sample(outputs["eventness_logits"], 1)
    event_types = _one_sample(outputs["event_type_logits"], 2)
    event_activity = _one_sample(outputs["event_activity_logits"], 2)
    pointers = _one_sample(outputs["track_pointer_logits"], 2)
    onset_logits = _one_sample(outputs.get("onset_logits", event_activity), 2)
    offset_logits = _one_sample(outputs.get("offset_logits", event_activity), 2)

    frame_count = track_activity.shape[-1]
    grid_duration = _grid_floor(duration_sec, frame_sec)
    if grid_duration <= 0:
        raise ValueError("duration_sec must contain at least one output frame")
    expected_frames = round(duration_sec / frame_sec)
    if abs(frame_count - expected_frames) > 1:
        raise ValueError(
            f"Feature grid has {frame_count} frames but duration implies {expected_frames}"
        )

    tracks: list[Track] = []
    slot_to_track: dict[int, str] = {}
    track_masks: dict[int, np.ndarray] = {}
    for slot, presence_logit in enumerate(track_presence):
        presence = _sigmoid(float(presence_logit))
        if presence < track_threshold:
            continue
        type_probability = _softmax(track_types[slot])
        type_index = int(np.argmax(type_probability))
        activity_probability = _sigmoid_array(track_activity[slot])
        activity_mask = activity_probability >= activity_threshold
        spans = _mask_spans(activity_mask, frame_sec, grid_duration)
        track_id = f"T{len(tracks) + 1}"
        slot_to_track[slot] = track_id
        track_masks[slot] = activity_mask
        confidence = float(presence * type_probability[type_index])
        tracks.append(
            Track(
                id=track_id,
                kind=TRACK_KINDS[type_index],  # type: ignore[arg-type]
                spans=spans,
                confidence=confidence,
                audibility=_sigmoid(float(track_audibility[slot])),
                evidence=Evidence(
                    method="track_event_slot_decoder",
                    spans=spans,
                    audio_support=confidence,
                ),
                attributes={"source_slot": slot},
            )
        )

    events: list[Event] = []
    for slot, eventness_logit in enumerate(eventness):
        presence = _sigmoid(float(eventness_logit))
        if presence < event_threshold:
            continue
        type_probability = _softmax(event_types[slot])
        type_index = int(np.argmax(type_probability))
        pointer_index = int(np.argmax(_softmax(pointers[slot])))
        track_id = slot_to_track.get(pointer_index)
        raw_mask = _sigmoid_array(event_activity[slot]) >= activity_threshold
        if pointer_index in track_masks:
            contained_mask = raw_mask & track_masks[pointer_index]
            if contained_mask.any():
                raw_mask = contained_mask
        spans = _mask_spans(raw_mask, frame_sec, grid_duration)
        if not spans:
            onset = int(np.argmax(onset_logits[slot]))
            offset = max(onset, int(np.argmax(offset_logits[slot])))
            spans = [
                Span(
                    round(onset * frame_sec, 6),
                    min(round((offset + 1) * frame_sec, 6), grid_duration),
                )
            ]
        event_type = EVENT_TYPES[type_index]
        confidence = float(presence * type_probability[type_index])
        text = _event_text(event_texts, slot, event_type)
        events.append(
            Event(
                id=f"E{len(events) + 1}",
                type=event_type,  # type: ignore[arg-type]
                track_id=track_id,
                spans=spans,
                text=text,
                confidence=confidence,
                evidence=Evidence(
                    method="track_event_slot_decoder",
                    spans=spans,
                    audio_support=confidence,
                ),
                attributes={"source_slot": slot, "pointer_slot": pointer_index},
            )
        )
    events.sort(key=lambda event: (event.onset, event.id))
    for index, event in enumerate(events, 1):
        event.id = f"E{index}"
    ledger = Ledger(
        sample_id=sample_id,
        duration_sec=duration_sec,
        tracks=tracks,
        events=events,
        provenance={"label_level": "model_prediction"},
    )
    ledger.validate()
    return ledger


def _one_sample(value: Any, unbatched_ndim: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == unbatched_ndim + 1:
        if array.shape[0] != 1:
            raise ValueError("decode_slot_arrays accepts exactly one sample")
        array = array[0]
    if array.ndim != unbatched_ndim:
        raise ValueError(f"Expected {unbatched_ndim} dimensions, got {array.shape}")
    return array


def _mask_spans(mask: np.ndarray, frame_sec: float, duration_sec: float) -> list[Span]:
    pairs = activity_to_spans(
        mask,
        frame_sec=frame_sec,
        merge_gap_sec=0.0,
        minimum_duration_sec=frame_sec,
    )
    return [Span(start, min(end, duration_sec)) for start, end in pairs if start < duration_sec]


def _event_text(
    event_texts: Sequence[str] | Mapping[int, str] | None, slot: int, event_type: str
) -> str:
    if isinstance(event_texts, Mapping):
        value = event_texts.get(slot)
    elif event_texts is not None and slot < len(event_texts):
        value = event_texts[slot]
    else:
        value = None
    return (
        str(value).strip()
        if value and str(value).strip()
        else f"untranscribed {event_type} event"
    )


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = np.asarray(value, dtype=np.float64) - float(np.max(value))
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _sigmoid_array(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _grid_floor(value: float, resolution_sec: float) -> float:
    frames = int(np.floor((value + 1e-8) / resolution_sec))
    return round(frames * resolution_sec, 6)
