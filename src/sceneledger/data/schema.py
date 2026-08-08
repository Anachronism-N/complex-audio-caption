"""Canonical track-event ledger schema (v0.2).

Pydantic mirror of ``schemas/track_event_ledger.schema.json``. The JSON schema is
the single source of truth for the on-disk canonical representation; this module
adds ergonomic Python objects plus cross-field validation that JSON Schema
cannot express on its own (track_id references, ID uniqueness, span ordering).

All times are quantized to ``time_resolution_sec = 0.1``. The serializer and
parser depend on these types; do not change field names without bumping the
schema version.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.2.0"
TIME_RESOLUTION_SEC = 0.1
TIME_GRID_TOLERANCE = 1e-6

TrackKind = Literal["speech", "vocal", "music", "sfx", "ambience", "residual"]
EventType = Literal["speech", "lys", "music", "sfx"]
RelationPredicate = Literal[
    "before", "after", "overlaps", "interrupts", "causes", "echo_of"
]
LabelLevel = Literal["A", "B", "C", "D", "E", "human", "model_prediction"]

_TRACK_ID_RE = re.compile(r"^T[0-9]+$")
_EVENT_ID_RE = re.compile(r"^E[0-9]+$")


def _quantize(value: float) -> float:
    """Snap a float to the 0.1 s grid, tolerating float noise."""
    snapped = round(value / TIME_RESOLUTION_SEC) * TIME_RESOLUTION_SEC
    # avoid -0.0 and trailing float dust
    return round(snapped, 6)


class Span(BaseModel):
    """A single ``[start, end]`` interval, quantized to 0.1 s."""

    model_config = ConfigDict(extra="forbid")

    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., gt=0.0)
    start_uncertainty_sec: float | None = Field(default=None, ge=0.0)
    end_uncertainty_sec: float | None = Field(default=None, ge=0.0)

    @field_validator("start_sec", "end_sec")
    @classmethod
    def _grid(cls, v: float) -> float:
        return _quantize(v)

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.end_sec <= self.start_sec:
            raise ValueError(
                f"span end_sec ({self.end_sec}) must be strictly greater than "
                f"start_sec ({self.start_sec})"
            )
        # uncertainties cannot extend span below 0
        if self.start_uncertainty_sec is not None and self.start_uncertainty_sec < 0:
            raise ValueError("start_uncertainty_sec must be >= 0")
        if self.end_uncertainty_sec is not None and self.end_uncertainty_sec < 0:
            raise ValueError("end_uncertainty_sec must be >= 0")
        return self

    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 6)


class Evidence(BaseModel):
    """Acoustic / audiovisual evidence backing a track or event."""

    model_config = ConfigDict(extra="forbid")

    method: str | None = None
    audio_support: float | None = Field(default=None, ge=0.0, le=1.0)
    target_residual_margin: float | None = None
    av_support: float | None = Field(default=None, ge=0.0, le=1.0)
    waveform_uri: str | None = None
    mask_uri: str | None = None


class Track(BaseModel):
    """A persistent source lane (speaker / singer / music / sfx / ambience)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: TrackKind
    identity: str | None = None
    spans: list[Span] = Field(default_factory=list)
    audibility: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence | None = None
    attributes: dict[str, str | float | bool | None] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not _TRACK_ID_RE.match(v):
            raise ValueError(f"track id must match {_TRACK_ID_RE.pattern}; got {v!r}")
        return v

    @model_validator(mode="after")
    def _spans_disjoint(self) -> Track:
        spans = sorted(self.spans, key=lambda s: s.start_sec)
        for prev, cur in zip(spans, spans[1:]):
            if cur.start_sec < prev.end_sec - TIME_GRID_TOLERANCE:
                raise ValueError(
                    f"track {self.id} has overlapping spans: "
                    f"[{prev.start_sec},{prev.end_sec}] vs [{cur.start_sec},{cur.end_sec}]"
                )
        return self


class Relation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicate: RelationPredicate
    target_event_id: str


class Event(BaseModel):
    """The minimal caption unit produced by a track."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: EventType
    track_id: str | None = None
    spans: list[Span] = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    verbatim: bool | None = None
    language: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence | None = None
    relations: list[Relation] = Field(default_factory=list)
    attributes: dict[str, str | float | bool | None] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not _EVENT_ID_RE.match(v):
            raise ValueError(f"event id must match {_EVENT_ID_RE.pattern}; got {v!r}")
        return v

    @model_validator(mode="after")
    def _spans_disjoint(self) -> Event:
        spans = sorted(self.spans, key=lambda s: s.start_sec)
        for prev, cur in zip(spans, spans[1:]):
            if cur.start_sec < prev.end_sec - TIME_GRID_TOLERANCE:
                raise ValueError(
                    f"event {self.id} has overlapping spans: "
                    f"[{prev.start_sec},{prev.end_sec}] vs [{cur.start_sec},{cur.end_sec}]"
                )
        return self

    def start_sec(self) -> float:
        return min(s.start_sec for s in self.spans)

    def end_sec(self) -> float:
        return max(s.end_sec for s in self.spans)


class Conditions(BaseModel):
    """Acoustic conditions of the clip (reported, not required)."""

    model_config = ConfigDict(extra="forbid")

    domain: str | None = None
    snr_db: float | None = None
    t60_sec: float | None = Field(default=None, ge=0.0)
    echo: bool | None = None
    codec: str | None = None
    overlap_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class Provenance(BaseModel):
    """Where the ledger came from."""

    model_config = ConfigDict(extra="forbid")

    label_level: LabelLevel | None = None
    source_dataset: str | None = None
    renderer_manifest_uri: str | None = None
    teacher_versions: dict[str, str] = Field(default_factory=dict)
    license_status: str | None = None


class Ledger(BaseModel):
    """The canonical track-event representation of one clip."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    sample_id: str = Field(..., min_length=1)
    duration_sec: float = Field(..., gt=0.0)
    time_resolution_sec: Literal[TIME_RESOLUTION_SEC] = TIME_RESOLUTION_SEC
    language: str | None = None
    conditions: Conditions = Field(default_factory=Conditions)
    tracks: list[Track] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("duration_sec")
    @classmethod
    def _grid_duration(cls, v: float) -> float:
        return _quantize(v)

    @model_validator(mode="after")
    def _cross_refs(self) -> Ledger:
        track_ids = [t.id for t in self.tracks]
        event_ids = [e.id for e in self.events]

        dup_tracks = [k for k, c in Counter(track_ids).items() if c > 1]
        if dup_tracks:
            raise ValueError(f"duplicate track ids: {dup_tracks}")
        dup_events = [k for k, c in Counter(event_ids).items() if c > 1]
        if dup_events:
            raise ValueError(f"duplicate event ids: {dup_events}")

        track_id_set = set(track_ids)
        for e in self.events:
            if e.track_id is not None and e.track_id not in track_id_set:
                raise ValueError(
                    f"event {e.id} references unknown track_id {e.track_id!r}"
                )

        # events must stay within clip duration
        for e in self.events:
            if e.end_sec() > self.duration_sec + TIME_GRID_TOLERANCE:
                raise ValueError(
                    f"event {e.id} extends past duration_sec "
                    f"({e.end_sec()} > {self.duration_sec})"
                )
            if e.start_sec() < -TIME_GRID_TOLERANCE:
                raise ValueError(f"event {e.id} starts before 0 ({e.start_sec()})")
        for t in self.tracks:
            for s in t.spans:
                if s.end_sec > self.duration_sec + TIME_GRID_TOLERANCE:
                    raise ValueError(
                        f"track {t.id} span extends past duration_sec "
                        f"({s.end_sec} > {self.duration_sec})"
                    )

        # relation targets must exist
        event_id_set = set(event_ids)
        for e in self.events:
            for r in e.relations:
                if r.target_event_id not in event_id_set:
                    raise ValueError(
                        f"event {e.id} relation targets unknown event "
                        f"{r.target_event_id!r}"
                    )
        return self

    def event_count(self) -> int:
        return len(self.events)

    def track_count(self) -> int:
        return len(self.tracks)
