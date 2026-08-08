from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..types import EventType, Span, TrackKind


@dataclass
class TeacherContext:
    sample_id: str
    duration_sec: float
    round_index: int = 0
    accepted_tracks: list[dict[str, Any]] = field(default_factory=list)
    accepted_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrackProposal:
    kind: TrackKind
    spans: list[Span]
    confidence: float
    proposer: str
    identity: str | None = None
    waveform_uri: str | None = None
    attributes: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass
class CaptionCandidate:
    type: EventType
    spans: list[Span]
    text: str
    confidence: float
    captioner: str
    verbatim: bool | None = None
    language: str | None = None


@dataclass
class Verification:
    accepted: bool
    audio_support: float
    verifier: str
    reason: str = ""
    corrected_text: str | None = None
    corrected_spans: list[Span] | None = None


class TrackProposer(Protocol):
    name: str

    def propose(self, audio_path: str, context: TeacherContext) -> list[TrackProposal]: ...


class TrackCaptioner(Protocol):
    name: str

    def caption(
        self, audio_path: str, proposal: TrackProposal, context: TeacherContext
    ) -> list[CaptionCandidate]: ...


class CaptionVerifier(Protocol):
    name: str

    def verify(
        self,
        audio_path: str,
        proposal: TrackProposal,
        candidate: CaptionCandidate,
        context: TeacherContext,
    ) -> Verification: ...
