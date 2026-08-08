from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

EventType = Literal["speech", "lys", "music", "sfx"]
TrackKind = Literal["speech", "vocal", "music", "sfx", "ambience", "residual"]


@dataclass(frozen=True, order=True)
class Span:
    start_sec: float
    end_sec: float
    start_uncertainty_sec: float | None = None
    end_uncertainty_sec: float | None = None

    def validate(self, duration_sec: float | None = None, resolution_sec: float = 0.1) -> None:
        values = (self.start_sec, self.end_sec)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite span: {self}")
        if self.start_sec < 0 or self.end_sec <= self.start_sec:
            raise ValueError(f"Invalid span: {self}")
        if duration_sec is not None and self.end_sec > duration_sec + 1e-6:
            raise ValueError(f"Span {self} exceeds duration {duration_sec}")
        for name, value in (
            ("start_uncertainty_sec", self.start_uncertainty_sec),
            ("end_uncertainty_sec", self.end_uncertainty_sec),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid {name}: {value}")
        if resolution_sec > 0:
            for value in values:
                if abs(value / resolution_sec - round(value / resolution_sec)) > 1e-5:
                    raise ValueError(f"{value} is not aligned to {resolution_sec}s")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Span:
        return cls(
            start_sec=float(value["start_sec"]),
            end_sec=float(value["end_sec"]),
            start_uncertainty_sec=_optional_float(value.get("start_uncertainty_sec")),
            end_uncertainty_sec=_optional_float(value.get("end_uncertainty_sec")),
        )


@dataclass
class Evidence:
    method: str | None = None
    spans: list[Span] = field(default_factory=list)
    audio_support: float | None = None
    target_residual_margin: float | None = None
    av_support: float | None = None
    waveform_uri: str | None = None
    mask_uri: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> Evidence | None:
        if value is None:
            return None
        return cls(
            method=value.get("method"),
            spans=[Span.from_dict(item) for item in value.get("spans", [])],
            audio_support=_optional_float(value.get("audio_support")),
            target_residual_margin=_optional_float(value.get("target_residual_margin")),
            av_support=_optional_float(value.get("av_support")),
            waveform_uri=value.get("waveform_uri"),
            mask_uri=value.get("mask_uri"),
        )


@dataclass
class Track:
    id: str
    kind: TrackKind
    spans: list[Span]
    confidence: float
    identity: str | None = None
    audibility: float | None = None
    evidence: Evidence | None = None
    attributes: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Track:
        return cls(
            id=str(value["id"]),
            kind=value["kind"],
            spans=[Span.from_dict(item) for item in value.get("spans", [])],
            confidence=float(value["confidence"]),
            identity=value.get("identity"),
            audibility=_optional_float(value.get("audibility")),
            evidence=Evidence.from_dict(value.get("evidence")),
            attributes=dict(value.get("attributes", {})),
        )


@dataclass
class Event:
    id: str
    type: EventType
    track_id: str | None
    spans: list[Span]
    text: str
    confidence: float
    verbatim: bool | None = None
    language: str | None = None
    evidence: Evidence | None = None
    relations: list[dict[str, str]] = field(default_factory=list)
    attributes: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    @property
    def onset(self) -> float:
        return min(span.start_sec for span in self.spans)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Event:
        return cls(
            id=str(value["id"]),
            type=value["type"],
            track_id=value.get("track_id"),
            spans=[Span.from_dict(item) for item in value["spans"]],
            text=str(value["text"]),
            confidence=float(value["confidence"]),
            verbatim=value.get("verbatim"),
            language=value.get("language"),
            evidence=Evidence.from_dict(value.get("evidence")),
            relations=list(value.get("relations", [])),
            attributes=dict(value.get("attributes", {})),
        )


@dataclass
class Ledger:
    sample_id: str
    duration_sec: float
    tracks: list[Track]
    events: list[Event]
    schema_version: str = "0.2.0"
    time_resolution_sec: float = 0.1
    language: str | None = None
    conditions: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not math.isfinite(self.duration_sec) or self.duration_sec <= 0:
            raise ValueError("duration_sec must be positive and finite")
        if self.schema_version != "0.2.0":
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        if abs(self.time_resolution_sec - 0.1) > 1e-8:
            raise ValueError("SceneLedger v0.2 requires a 0.1 second output grid")

        track_ids = [track.id for track in self.tracks]
        event_ids = [event.id for event in self.events]
        _require_unique(track_ids, "track")
        _require_unique(event_ids, "event")
        track_id_set = set(track_ids)

        for track in self.tracks:
            _validate_confidence(track.confidence, f"track {track.id}")
            if track.audibility is not None:
                _validate_confidence(track.audibility, f"track {track.id} audibility")
            _validate_spans(track.spans, self.duration_sec, self.time_resolution_sec, False)
            _validate_evidence(
                track.evidence, self.duration_sec, self.time_resolution_sec, f"track {track.id}"
            )

        for event in self.events:
            _validate_confidence(event.confidence, f"event {event.id}")
            if event.track_id is not None and event.track_id not in track_id_set:
                raise ValueError(f"Event {event.id} references unknown track {event.track_id}")
            if not event.text.strip():
                raise ValueError(f"Event {event.id} has empty text")
            _validate_spans(event.spans, self.duration_sec, self.time_resolution_sec, True)
            _validate_evidence(
                event.evidence, self.duration_sec, self.time_resolution_sec, f"event {event.id}"
            )
            for relation in event.relations:
                target = relation.get("target_event_id")
                if target not in set(event_ids):
                    raise ValueError(f"Event {event.id} relation references {target!r}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return _drop_none(value)

    @classmethod
    def from_dict(cls, value: dict[str, Any], validate: bool = True) -> Ledger:
        ledger = cls(
            schema_version=str(value.get("schema_version", "0.2.0")),
            sample_id=str(value["sample_id"]),
            duration_sec=float(value["duration_sec"]),
            time_resolution_sec=float(value.get("time_resolution_sec", 0.1)),
            language=value.get("language"),
            conditions=value.get("conditions"),
            provenance=value.get("provenance"),
            tracks=[Track.from_dict(item) for item in value.get("tracks", [])],
            events=[Event.from_dict(item) for item in value.get("events", [])],
        )
        if validate:
            ledger.validate()
        return ledger


def read_jsonl(path: str | Path, validate: bool = True) -> Iterator[Ledger]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield Ledger.from_dict(json.loads(line), validate=validate)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def write_jsonl(path: str | Path, ledgers: Iterable[Ledger]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for ledger in ledgers:
            ledger.validate()
            handle.write(json.dumps(ledger.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def quantize_time(value: float, resolution_sec: float = 0.1) -> float:
    return round(round(value / resolution_sec) * resolution_sec, 6)


def _validate_spans(
    spans: list[Span], duration_sec: float, resolution_sec: float, require_nonempty: bool
) -> None:
    if require_nonempty and not spans:
        raise ValueError("At least one span is required")
    previous_end = -math.inf
    for span in spans:
        span.validate(duration_sec, resolution_sec)
        if span.start_sec < previous_end - 1e-6:
            raise ValueError(f"Spans overlap or are unsorted: {spans}")
        previous_end = span.end_sec


def _require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} IDs: {values}")


def _validate_confidence(value: float, label: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} confidence must be in [0, 1], got {value}")


def _validate_evidence(
    evidence: Evidence | None, duration_sec: float, resolution_sec: float, label: str
) -> None:
    if evidence is None:
        return
    _validate_spans(evidence.spans, duration_sec, resolution_sec, False)
    for name, value in (
        ("audio_support", evidence.audio_support),
        ("av_support", evidence.av_support),
    ):
        if value is not None:
            _validate_confidence(value, f"{label} evidence {name}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value
