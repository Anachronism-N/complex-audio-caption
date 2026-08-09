"""Compatibility wrapper for the superseded standalone S1a-v2 decoder.

Boundary regression now lives beside the activity head in :mod:`event_slots`
so both decoders share data, matching, checkpoints, and evaluation. New
experiments should instantiate :class:`EventSlotDecoder` directly.
"""

from __future__ import annotations

import torch

from sceneledger.models.event_slots import (
    EVENT_TYPES,
    N_EVENT_TYPES,
    EventSlotDecoder,
)


class EventSlotDecoderV2(EventSlotDecoder):
    """Backward-compatible boundary-only view of the unified dual-head model."""

    def __init__(
        self,
        feature_dim: int = 2560,
        hidden_dim: int = 768,
        n_slots: int = 8,
        n_heads: int = 8,
        n_layers: int = 6,
        max_duration_sec: float = 30.0,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_slots=n_slots,
            n_heads=n_heads,
            n_layers=n_layers,
            max_duration_sec=max_duration_sec,
            use_temporal_embedding=True,
        )

    def predict(
        self, audio_features: torch.Tensor, threshold: float = 0.4
    ) -> list[list[dict]]:
        events_by_sample = super().predict(
            audio_features,
            eventness_threshold=threshold,
            decode_mode="boundary",
        )
        return [
            [
                {
                    "type": event["type"],
                    "onset": event["spans"][0]["start_sec"],
                    "offset": event["spans"][-1]["end_sec"],
                    "confidence": event["confidence"],
                }
                for event in events
            ]
            for events in events_by_sample
        ]


__all__ = ["EVENT_TYPES", "EventSlotDecoderV2", "N_EVENT_TYPES"]
