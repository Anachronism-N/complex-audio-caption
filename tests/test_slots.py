from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sceneledger.models.losses import slot_set_loss  # noqa: E402
from sceneledger.models.slots import TrackEventSlotConfig, TrackEventSlotDecoder  # noqa: E402


def test_slot_decoder_shapes_and_loss() -> None:
    config = TrackEventSlotConfig(
        input_dim=16,
        model_dim=32,
        track_slots=3,
        event_slots=4,
        attention_heads=4,
        decoder_layers=1,
        feedforward_dim=64,
        source_embedding_dim=8,
    )
    model = TrackEventSlotDecoder(config)
    outputs = model(torch.randn(1, 20, 16))
    assert outputs["track_activity_logits"].shape == (1, 3, 20)
    assert outputs["event_activity_logits"].shape == (1, 4, 20)
    assert outputs["track_pointer_logits"].shape == (1, 4, 4)
    target = {
        "track_type": torch.tensor([0, 2]),
        "track_activity": torch.tensor(
            [[1] * 10 + [0] * 10, [0] * 8 + [1] * 12], dtype=torch.float32
        ),
        "event_type": torch.tensor([0, 3]),
        "event_activity": torch.tensor(
            [[1] * 6 + [0] * 14, [0] * 8 + [1] * 5 + [0] * 7], dtype=torch.float32
        ),
        "event_track": torch.tensor([0, 1]),
    }
    losses = slot_set_loss(outputs, [target])
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
