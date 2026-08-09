"""Event slot decoder for S1 (SceneLedger core contribution).

A lightweight DETR-like set predictor on top of frozen MOSS audio encoder
features. K learned event queries cross-attend to temporal features and each
predict: eventness (null/active), type (speech/lys/music/sfx), and a 100ms
activity mask. Boundaries are derived from the activity mask (first/last
active frame), avoiding regression instability.

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
    ):
        super().__init__()
        self.n_slots = n_slots
        self.hidden_dim = hidden_dim
        self.max_frames = int(round(max_duration_sec / TIME_RESOLUTION_SEC))  # 300

        # project frozen features to hidden_dim
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # learned event queries
        self.query_embed = nn.Embedding(n_slots, hidden_dim)

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
        B = audio_features.shape[0]
        T_feat = audio_features.shape[1]

        # project features
        memory = self.input_proj(audio_features)  # [B, T_feat, hidden]

        # interpolate 12.5 Hz features to 10 Hz (100ms grid)
        # T_feat at 12.5 Hz -> T_100 at 10 Hz
        T_100 = min(self.max_frames, int(round(T_feat * 0.8)))  # 12.5->10 Hz ratio
        if T_feat != T_100:
            memory = memory.transpose(1, 2)  # [B, hidden, T_feat]
            memory = F.interpolate(memory, size=T_100, mode="linear", align_corners=False)
            memory = memory.transpose(1, 2)  # [B, T_100, hidden]

        # expand queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, K, hidden]

        # cross-attention: queries attend to memory
        # TransformerDecoder expects tgt=queries, memory=features
        if feature_mask is not None:
            # interpolate feature_mask to T_100
            if feature_mask.shape[1] != T_100:
                feature_mask = F.interpolate(
                    feature_mask.float().unsqueeze(1), size=T_100, mode="nearest"
                ).squeeze(1).bool()

        tgt = queries
        decoded = self.decoder(tgt, memory, memory_key_padding_mask=~feature_mask if feature_mask is not None else None)

        # prediction heads
        eventness_logits = self.eventness_head(decoded).squeeze(-1)  # [B, K]
        type_logits = self.type_head(decoded)  # [B, K, 4]
        activity_logits = self.activity_head(decoded)  # [B, K, max_frames]

        # truncate activity to T_100
        activity_logits = activity_logits[:, :, :T_100]

        return {
            "eventness_logits": eventness_logits,
            "type_logits": type_logits,
            "activity_logits": activity_logits,
            "n_frames": T_100,
        }

    def predict(self, audio_features: torch.Tensor, feature_mask: torch.Tensor | None = None) -> list[dict]:
        """Decode slot outputs into a list of predicted events (for evaluation).

        Returns list of {type, activity_mask, onset, offset} per active slot.
        """
        out = self.forward(audio_features, feature_mask)
        eventness = out["eventness_logits"].sigmoid()
        type_probs = out["type_logits"].softmax(-1)
        activity = out["activity_logits"].sigmoid()

        B, K = eventness.shape
        results: list[list[dict]] = []
        for b in range(B):
            events = []
            for k in range(K):
                if eventness[b, k] < 0.4:  # threshold tuned to balance precision/recall
                    continue
                t_idx = type_probs[b, k].argmax().item()
                etype = EVENT_TYPES[t_idx]
                act = activity[b, k] > 0.5
                # derive onset/offset from activity
                active_indices = act.nonzero(as_tuple=True)[0]
                if len(active_indices) == 0:
                    continue
                onset = active_indices[0].item() * TIME_RESOLUTION_SEC
                offset = (active_indices[-1].item() + 1) * TIME_RESOLUTION_SEC
                events.append({
                    "type": etype,
                    "onset": round(onset, 1),
                    "offset": round(offset, 1),
                    "activity_mask": act.cpu().numpy(),
                    "confidence": round(eventness[b, k].item(), 3),
                })
            results.append(events)
        return results


__all__ = ["EVENT_TYPES", "EventSlotDecoder", "N_EVENT_TYPES"]
