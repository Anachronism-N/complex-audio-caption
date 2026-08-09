"""Event slot decoder v2: boundary regression instead of activity mask.

Simpler and more stable than v1: each slot predicts eventness, type, and
onset/offset (continuous regression) instead of a full activity mask.
Uses L1 loss for boundaries, which converges faster than Dice on masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sceneledger.data.schema import TIME_RESOLUTION_SEC

EVENT_TYPES = ("speech", "lys", "music", "sfx")
N_EVENT_TYPES = 4


class EventSlotDecoderV2(nn.Module):
    """K event queries → eventness / type / onset / offset via cross-attention."""

    def __init__(
        self,
        feature_dim: int = 2560,
        hidden_dim: int = 768,
        n_slots: int = 8,
        n_heads: int = 8,
        n_layers: int = 6,
        max_duration_sec: float = 30.0,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.hidden_dim = hidden_dim
        self.max_duration_sec = max_duration_sec

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.query_embed = nn.Embedding(n_slots, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.eventness_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, N_EVENT_TYPES)
        # boundary regression: onset and offset in [0, max_duration]
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [onset, offset]
        )

    def forward(self, audio_features: torch.Tensor) -> dict[str, torch.Tensor]:
        B = audio_features.shape[0]
        T_feat = audio_features.shape[1]

        memory = self.input_proj(audio_features)  # [B, T_feat, hidden]

        # interpolate 12.5 Hz -> 10 Hz for consistent temporal resolution
        T_100 = int(round(T_feat * 0.8))
        if T_feat != T_100:
            memory = memory.transpose(1, 2)
            memory = F.interpolate(memory, size=T_100, mode="linear", align_corners=False)
            memory = memory.transpose(1, 2)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, memory)

        eventness_logits = self.eventness_head(decoded).squeeze(-1)  # [B, K]
        type_logits = self.type_head(decoded)  # [B, K, 4]
        # sigmoid -> [0, max_duration]
        boundary_raw = self.boundary_head(decoded)  # [B, K, 2]
        boundary = torch.sigmoid(boundary_raw) * self.max_duration_sec  # [B, K, 2]

        return {
            "eventness_logits": eventness_logits,
            "type_logits": type_logits,
            "onset": boundary[..., 0],  # [B, K]
            "offset": boundary[..., 1],  # [B, K]
            "n_frames": T_100,
        }

    def predict(self, audio_features: torch.Tensor, threshold: float = 0.4) -> list[list[dict]]:
        out = self.forward(audio_features)
        eventness = out["eventness_logits"].sigmoid()
        type_probs = out["type_logits"].softmax(-1)
        onsets = out["onset"]
        offsets = out["offset"]

        B, K = eventness.shape
        results = []
        for b in range(B):
            events = []
            for k in range(K):
                if eventness[b, k] < threshold:
                    continue
                t_idx = type_probs[b, k].argmax().item()
                onset = onsets[b, k].item()
                offset = offsets[b, k].item()
                if offset <= onset:
                    continue
                events.append({
                    "type": EVENT_TYPES[t_idx],
                    "onset": round(onset, 1),
                    "offset": round(offset, 1),
                    "confidence": round(eventness[b, k].item(), 3),
                })
            results.append(events)
        return results


__all__ = ["EVENT_TYPES", "EventSlotDecoderV2", "N_EVENT_TYPES"]
