from __future__ import annotations

from sceneledger.teachers import (
    CaptionCandidate,
    TeacherPipeline,
    TeacherPipelineConfig,
    TrackProposal,
    Verification,
)
from sceneledger.types import Span


class FakeProposer:
    name = "fake_proposer"

    def propose(self, audio_path, context):
        if context.round_index > 0:
            return []
        return [
            TrackProposal("speech", [Span(0.0, 1.0)], 0.9, self.name),
            TrackProposal("sfx", [Span(1.0, 1.5)], 0.8, self.name),
        ]


class FakeCaptioner:
    name = "fake_captioner"

    def caption(self, audio_path, proposal, context):
        event_type = "speech" if proposal.kind == "speech" else "sfx"
        return [
            CaptionCandidate(
                event_type,
                proposal.spans,
                "hello" if event_type == "speech" else "hallucinated bang",
                0.9,
                self.name,
            )
        ]


class FakeVerifier:
    name = "fake_verifier"

    def verify(self, audio_path, proposal, candidate, context):
        accepted = candidate.type == "speech"
        return Verification(accepted, 0.85 if accepted else 0.1, self.name)


def test_teacher_pipeline_rejects_unsupported_candidate_and_stops() -> None:
    pipeline = TeacherPipeline(
        [FakeProposer()],
        [FakeCaptioner()],
        [FakeVerifier()],
        TeacherPipelineConfig(max_rounds=3),
    )
    ledger = pipeline.run("teacher", "unused.wav", 2.0)
    assert len(ledger.tracks) == 1
    assert len(ledger.events) == 1
    assert ledger.events[0].text == "hello"
    assert ledger.events[0].evidence.audio_support == 0.85
