from __future__ import annotations

from dataclasses import dataclass

from ..matching import temporal_iou
from ..types import Event, Evidence, Ledger, Track
from .base import (
    CaptionCandidate,
    CaptionVerifier,
    TeacherContext,
    TrackCaptioner,
    TrackProposal,
    TrackProposer,
)


@dataclass(frozen=True)
class TeacherPipelineConfig:
    max_rounds: int = 3
    max_tracks: int = 10
    proposal_dedup_iou: float = 0.8
    minimum_audio_support: float = 0.5
    require_all_verifiers: bool = True


class TeacherPipeline:
    """Plug-in orchestration for separation/diarization, captioning and verification.

    Adapters for SAM-Audio, diarization, ASR, MER and VLM teachers implement the
    small protocols in ``base.py``. The orchestrator itself remains CPU-testable.
    """

    def __init__(
        self,
        proposers: list[TrackProposer],
        captioners: list[TrackCaptioner],
        verifiers: list[CaptionVerifier],
        config: TeacherPipelineConfig | None = None,
    ) -> None:
        if not proposers or not captioners:
            raise ValueError("At least one proposer and captioner are required")
        self.proposers = proposers
        self.captioners = captioners
        self.verifiers = verifiers
        self.config = config or TeacherPipelineConfig()

    def run(self, sample_id: str, audio_path: str, duration_sec: float) -> Ledger:
        context = TeacherContext(sample_id=sample_id, duration_sec=duration_sec)
        accepted_proposals: list[TrackProposal] = []
        accepted_candidates: list[tuple[int, CaptionCandidate, float, list[str]]] = []
        for round_index in range(self.config.max_rounds):
            context.round_index = round_index
            new_in_round = 0
            proposals = [
                proposal
                for proposer in self.proposers
                for proposal in proposer.propose(audio_path, context)
            ]
            proposals.sort(key=lambda item: item.confidence, reverse=True)
            for proposal in proposals:
                if len(accepted_proposals) >= self.config.max_tracks:
                    break
                if self._is_duplicate(proposal, accepted_proposals):
                    continue
                verified = self._caption_and_verify(audio_path, proposal, context)
                if not verified:
                    continue
                proposal_index = len(accepted_proposals)
                accepted_proposals.append(proposal)
                for candidate, support, verifier_names in verified:
                    accepted_candidates.append(
                        (proposal_index, candidate, support, verifier_names)
                    )
                context.accepted_tracks.append(
                    {"kind": proposal.kind, "spans": proposal.spans, "proposer": proposal.proposer}
                )
                context.accepted_events.extend(
                    {"type": item[0].type, "spans": item[0].spans, "text": item[0].text}
                    for item in verified
                )
                new_in_round += 1
            if new_in_round == 0 or len(accepted_proposals) >= self.config.max_tracks:
                break

        tracks = [
            Track(
                id=f"T{index + 1}",
                kind=proposal.kind,
                spans=proposal.spans,
                confidence=proposal.confidence,
                identity=proposal.identity,
                evidence=Evidence(
                    method=proposal.proposer,
                    spans=proposal.spans,
                    audio_support=proposal.confidence,
                    waveform_uri=proposal.waveform_uri,
                ),
                attributes=proposal.attributes,
            )
            for index, proposal in enumerate(accepted_proposals)
        ]
        events = []
        for event_index, (proposal_index, candidate, support, verifiers) in enumerate(
            accepted_candidates, 1
        ):
            events.append(
                Event(
                    id=f"E{event_index}",
                    type=candidate.type,
                    track_id=f"T{proposal_index + 1}",
                    spans=candidate.spans,
                    text=candidate.text,
                    confidence=min(candidate.confidence, support),
                    verbatim=candidate.verbatim,
                    language=candidate.language,
                    evidence=Evidence(
                        method="cross_teacher_verification",
                        spans=candidate.spans,
                        audio_support=support,
                    ),
                    attributes={
                        "captioner": candidate.captioner,
                        "verifiers": ",".join(verifiers),
                    },
                )
            )
        events.sort(key=lambda event: (event.onset, event.id))
        for index, event in enumerate(events, 1):
            event.id = f"E{index}"
        teacher_versions = {
            teacher.name: teacher.__class__.__name__
            for teacher in [*self.proposers, *self.captioners, *self.verifiers]
        }
        ledger = Ledger(
            sample_id=sample_id,
            duration_sec=duration_sec,
            tracks=tracks,
            events=events,
            provenance={"label_level": "C", "teacher_versions": teacher_versions},
        )
        ledger.validate()
        return ledger

    def _caption_and_verify(
        self, audio_path: str, proposal: TrackProposal, context: TeacherContext
    ) -> list[tuple[CaptionCandidate, float, list[str]]]:
        results: list[tuple[CaptionCandidate, float, list[str]]] = []
        candidates = [
            candidate
            for captioner in self.captioners
            for candidate in captioner.caption(audio_path, proposal, context)
        ]
        for candidate in candidates:
            decisions = [
                verifier.verify(audio_path, proposal, candidate, context)
                for verifier in self.verifiers
            ]
            accepted_flags = [decision.accepted for decision in decisions]
            accepted = (
                all(accepted_flags)
                if self.config.require_all_verifiers
                else any(accepted_flags) or not decisions
            )
            if self.config.require_all_verifiers and not decisions:
                accepted = True
            support = min(
                [candidate.confidence, *[decision.audio_support for decision in decisions]]
            )
            if not accepted or support < self.config.minimum_audio_support:
                continue
            for decision in decisions:
                if decision.corrected_text:
                    candidate.text = decision.corrected_text
                if decision.corrected_spans:
                    candidate.spans = decision.corrected_spans
            if not _spans_within(candidate, proposal):
                continue
            results.append((candidate, support, [decision.verifier for decision in decisions]))
        return results

    def _is_duplicate(
        self, proposal: TrackProposal, accepted: list[TrackProposal]
    ) -> bool:
        return any(
            proposal.kind == existing.kind
            and temporal_iou(proposal.spans, existing.spans) >= self.config.proposal_dedup_iou
            for existing in accepted
        )


def _spans_within(candidate: CaptionCandidate, proposal: TrackProposal) -> bool:
    return bool(candidate.spans) and all(
        any(
            track_span.start_sec <= event_span.start_sec + 1e-6
            and event_span.end_sec <= track_span.end_sec + 1e-6
            for track_span in proposal.spans
        )
        for event_span in candidate.spans
    )
