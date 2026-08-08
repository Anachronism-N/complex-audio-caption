from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrackEventSlotConfig:
    input_dim: int = 1024
    model_dim: int = 768
    track_slots: int = 8
    event_slots: int = 24
    track_types: int = 6
    event_types: int = 4
    attention_heads: int = 8
    decoder_layers: int = 3
    feedforward_dim: int = 2048
    source_embedding_dim: int = 256
    dropout: float = 0.1


class TrackEventSlotDecoder(nn.Module):
    """Two-level implicit decomposition over precomputed temporal audio features."""

    def __init__(self, config: TrackEventSlotConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.model_dim),
        )
        track_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        event_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.track_decoder = nn.TransformerDecoder(track_layer, config.decoder_layers)
        self.event_decoder = nn.TransformerDecoder(event_layer, config.decoder_layers)
        self.track_queries = nn.Parameter(torch.randn(config.track_slots, config.model_dim) * 0.02)
        self.event_queries = nn.Parameter(torch.randn(config.event_slots, config.model_dim) * 0.02)

        self.track_presence = nn.Linear(config.model_dim, 1)
        self.track_type = nn.Linear(config.model_dim, config.track_types)
        self.track_source_embedding = nn.Linear(config.model_dim, config.source_embedding_dim)
        self.track_audibility = nn.Linear(config.model_dim, 1)
        self.track_activity_query = nn.Linear(config.model_dim, config.model_dim)
        self.track_activity_key = nn.Linear(config.model_dim, config.model_dim)

        self.eventness = nn.Linear(config.model_dim, 1)
        self.event_type = nn.Linear(config.model_dim, config.event_types)
        self.event_activity_query = nn.Linear(config.model_dim, config.model_dim)
        self.event_activity_key = nn.Linear(config.model_dim, config.model_dim)
        self.onset_query = nn.Linear(config.model_dim, config.model_dim)
        self.offset_query = nn.Linear(config.model_dim, config.model_dim)
        self.boundary_key = nn.Linear(config.model_dim, config.model_dim)
        self.pointer_event = nn.Linear(config.model_dim, config.model_dim)
        self.pointer_track = nn.Linear(config.model_dim, config.model_dim)
        self.null_pointer = nn.Linear(config.model_dim, 1)
        self.local_projection = nn.Linear(config.model_dim, config.model_dim)

    def forward(self, features: Tensor, padding_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Args: features [B,T,input_dim], padding_mask [B,T] where True means padding."""
        memory = self.input_projection(features)
        batch_size = memory.shape[0]
        track_queries = self.track_queries.unsqueeze(0).expand(batch_size, -1, -1)
        track_slots = self.track_decoder(
            track_queries, memory, memory_key_padding_mask=padding_mask
        )
        track_activity = _bilinear_time_logits(
            self.track_activity_query(track_slots), self.track_activity_key(memory)
        )

        event_queries = self.event_queries.unsqueeze(0).expand(batch_size, -1, -1)
        event_memory = torch.cat([memory, track_slots], dim=1)
        if padding_mask is not None:
            track_padding = torch.zeros(
                batch_size, self.config.track_slots, dtype=torch.bool, device=padding_mask.device
            )
            event_padding = torch.cat([padding_mask, track_padding], dim=1)
        else:
            event_padding = None
        event_slots = self.event_decoder(
            event_queries, event_memory, memory_key_padding_mask=event_padding
        )
        event_activity = _bilinear_time_logits(
            self.event_activity_query(event_slots), self.event_activity_key(memory)
        )
        pointer_logits = torch.einsum(
            "bed,btd->bet",
            self.pointer_event(event_slots),
            self.pointer_track(track_slots),
        ) / math.sqrt(self.config.model_dim)
        pointer_logits = torch.cat([pointer_logits, self.null_pointer(event_slots)], dim=-1)
        boundary_keys = self.boundary_key(memory)
        onset_logits = _bilinear_time_logits(self.onset_query(event_slots), boundary_keys)
        offset_logits = _bilinear_time_logits(self.offset_query(event_slots), boundary_keys)

        weights = torch.sigmoid(event_activity)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask.unsqueeze(1), 0.0)
        local_feature = torch.einsum("bet,btd->bed", weights, memory)
        local_feature = local_feature / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return {
            "track_slots": track_slots,
            "track_presence_logits": self.track_presence(track_slots).squeeze(-1),
            "track_type_logits": self.track_type(track_slots),
            "track_activity_logits": track_activity,
            "track_source_embedding": nn.functional.normalize(
                self.track_source_embedding(track_slots), dim=-1
            ),
            "track_audibility_logits": self.track_audibility(track_slots).squeeze(-1),
            "event_slots": event_slots,
            "eventness_logits": self.eventness(event_slots).squeeze(-1),
            "event_type_logits": self.event_type(event_slots),
            "event_activity_logits": event_activity,
            "track_pointer_logits": pointer_logits,
            "onset_logits": onset_logits,
            "offset_logits": offset_logits,
            "local_feature": self.local_projection(local_feature),
        }


def containment_loss(
    event_activity_logits: Tensor, track_activity_logits: Tensor, pointers: Tensor
) -> Tensor:
    """Penalize event activity outside its assigned predicted track activity."""
    event_activity = torch.sigmoid(event_activity_logits)
    track_activity = torch.sigmoid(track_activity_logits)
    selected = torch.gather(
        track_activity,
        1,
        pointers.unsqueeze(-1).expand(-1, -1, track_activity.shape[-1]),
    )
    return torch.relu(event_activity - selected).mean()


def _bilinear_time_logits(queries: Tensor, keys: Tensor) -> Tensor:
    return torch.einsum("bsd,btd->bst", queries, keys) / math.sqrt(queries.shape[-1])
