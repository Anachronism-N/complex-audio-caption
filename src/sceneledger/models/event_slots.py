"""Event slot decoder for S1 (SceneLedger core contribution).

A lightweight DETR-like set predictor on top of frozen MOSS audio encoder
features. K learned event queries cross-attend to temporal features and each
predict: eventness (null/active), type (speech/lys/music/sfx), and a 100ms
activity mask. One event may contain multiple disjoint spans; decoding keeps
those spans instead of collapsing the mask to its first and last active frame.

This is S1a (event-only, no text). Text decoding (S1e) comes later by
conditioning the shared LLM on slot-local evidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sceneledger.data.schema import TIME_RESOLUTION_SEC

EVENT_TYPES = ("speech", "lys", "music", "sfx")
N_EVENT_TYPES = 4


class EventSlotDecoder(nn.Module):
    """K event queries → eventness / type / activity via cross-attention."""

    def __init__(
        self,
        feature_dim: int = 2560,
        hidden_dim: int = 768,
        n_slots: int = 24,
        n_heads: int = 8,
        n_layers: int = 4,
        max_duration_sec: float = 30.0,
        use_temporal_embedding: bool = True,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.hidden_dim = hidden_dim
        self.max_frames = int(round(max_duration_sec / TIME_RESOLUTION_SEC))  # 300

        # project frozen features to hidden_dim
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # learned event queries
        self.query_embed = nn.Embedding(n_slots, hidden_dim)
        self.temporal_embed = (
            nn.Embedding(self.max_frames, hidden_dim)
            if use_temporal_embedding
            else None
        )

        # transformer decoder (cross-attention from queries to features)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # prediction heads
        self.eventness_head = nn.Linear(hidden_dim, 1)  # null vs active
        self.type_head = nn.Linear(hidden_dim, N_EVENT_TYPES)
        self.activity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_frames),
        )

    def forward(
        self, audio_features: torch.Tensor, feature_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            audio_features: [B, T_feat, feature_dim] frozen MOSS encoder output.
            feature_mask: [B, T_feat] boolean mask for valid frames (padding).

        Returns:
            dict with eventness_logits [B, K], type_logits [B, K, 4],
            activity_logits [B, K, T_100].
        """
        batch_size = audio_features.shape[0]
        n_feature_frames = audio_features.shape[1]

        # project features
        memory = self.input_proj(audio_features)  # [B, T_feat, hidden]

        # interpolate 12.5 Hz features to 10 Hz (100ms grid)
        # T_feat at 12.5 Hz -> T_100 at 10 Hz
        n_activity_frames = max(
            1, min(self.max_frames, int(round(n_feature_frames * 0.8)))
        )
        if n_feature_frames != n_activity_frames:
            memory = memory.transpose(1, 2)  # [B, hidden, T_feat]
            memory = F.interpolate(
                memory, size=n_activity_frames, mode="linear", align_corners=False
            )
            memory = memory.transpose(1, 2)  # [B, T_100, hidden]
        if self.temporal_embed is not None:
            positions = torch.arange(n_activity_frames, device=memory.device)
            memory = memory + self.temporal_embed(positions).unsqueeze(0)

        # expand queries
        queries = self.query_embed.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, K, hidden]

        # cross-attention: queries attend to memory
        # TransformerDecoder expects tgt=queries, memory=features
        if feature_mask is not None:
            # interpolate feature_mask to T_100
            if feature_mask.shape[1] != n_activity_frames:
                feature_mask = F.interpolate(
                    feature_mask.float().unsqueeze(1),
                    size=n_activity_frames,
                    mode="nearest",
                ).squeeze(1).bool()

        tgt = queries
        memory_padding = ~feature_mask if feature_mask is not None else None
        decoded = self.decoder(tgt, memory, memory_key_padding_mask=memory_padding)

        # prediction heads
        eventness_logits = self.eventness_head(decoded).squeeze(-1)  # [B, K]
        type_logits = self.type_head(decoded)  # [B, K, 4]
        activity_logits = self.activity_head(decoded)  # [B, K, max_frames]

        # truncate activity to T_100
        activity_logits = activity_logits[:, :, :n_activity_frames]

        return {
            "eventness_logits": eventness_logits,
            "type_logits": type_logits,
            "activity_logits": activity_logits,
            "n_frames": n_activity_frames,
        }

    def predict(
        self,
        audio_features: torch.Tensor,
        feature_mask: torch.Tensor | None = None,
        *,
        eventness_threshold: float = 0.5,
        activity_threshold: float = 0.5,
    ) -> list[list[dict]]:
        """Decode slot outputs into a list of predicted events (for evaluation).

        Thresholds are explicit experiment parameters and must be recorded in
        the run metadata. Returns one event list per batch item.
        """
        out = self.forward(audio_features, feature_mask)
        eventness = out["eventness_logits"].sigmoid()
        type_probs = out["type_logits"].softmax(-1)
        activity = out["activity_logits"].sigmoid()

        batch_size, n_slots = eventness.shape
        results: list[list[dict]] = []
        for batch_index in range(batch_size):
            events = []
            for slot_index in range(n_slots):
                confidence = eventness[batch_index, slot_index].item()
                if confidence < eventness_threshold:
                    continue
                t_idx = type_probs[batch_index, slot_index].argmax().item()
                etype = EVENT_TYPES[t_idx]
                act = activity[batch_index, slot_index] >= activity_threshold
                spans = activity_mask_to_spans(act)
                if not spans:
                    continue
                events.append(
                    {
                        "type": etype,
                        "spans": spans,
                        "activity_mask": act.cpu().numpy(),
                        "confidence": round(confidence, 6),
                    }
                )
            results.append(events)
        return results


def activity_mask_to_spans(mask: torch.Tensor) -> list[dict[str, float]]:
    """Convert a boolean 100 ms mask into disjoint canonical spans."""
    active_indices = mask.to(dtype=torch.bool).nonzero(as_tuple=True)[0].tolist()
    if not active_indices:
        return []

    spans: list[dict[str, float]] = []
    start = previous = active_indices[0]
    for index in active_indices[1:]:
        if index != previous + 1:
            spans.append(
                {
                    "start_sec": round(start * TIME_RESOLUTION_SEC, 1),
                    "end_sec": round((previous + 1) * TIME_RESOLUTION_SEC, 1),
                }
            )
            start = index
        previous = index
    spans.append(
        {
            "start_sec": round(start * TIME_RESOLUTION_SEC, 1),
            "end_sec": round((previous + 1) * TIME_RESOLUTION_SEC, 1),
        }
    )
    return spans


__all__ = [
    "EVENT_TYPES",
    "EventSlotDecoder",
    "N_EVENT_TYPES",
    "activity_mask_to_spans",
]
