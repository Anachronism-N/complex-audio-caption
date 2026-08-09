"""Torch-gated unit tests for the S1a event-slot implementation."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sceneledger.losses.set_prediction import (  # noqa: E402
    _events_to_targets,
    set_prediction_loss,
)
from sceneledger.models.event_slots import (  # noqa: E402
    EventSlotDecoder,
    activity_mask_to_spans,
    boundary_to_span,
    hybridize_spans,
)


def test_event_target_preserves_disjoint_spans() -> None:
    targets = _events_to_targets(
        [
            {
                "type": "sfx",
                "spans": [
                    {"start_sec": 0.1, "end_sec": 0.3},
                    {"start_sec": 0.6, "end_sec": 0.8},
                ],
            }
        ],
        n_frames=10,
        max_slots=4,
    )
    assert targets["n_events"] == 1
    assert targets["activity_targets"][0].tolist() == [
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    assert targets["boundary_targets"][0].tolist() == pytest.approx([0.1, 0.8])


def test_activity_decoder_does_not_fill_gaps() -> None:
    mask = torch.tensor([False, True, True, False, False, True, False])
    assert activity_mask_to_spans(mask) == [
        {"start_sec": 0.1, "end_sec": 0.3},
        {"start_sec": 0.5, "end_sec": 0.6},
    ]


def test_hybrid_decoder_clips_activity_but_preserves_internal_gap() -> None:
    boundary = boundary_to_span(0.14, 0.76, 1.0)
    assert boundary == {"start_sec": 0.1, "end_sec": 0.8}
    assert hybridize_spans(
        [
            {"start_sec": 0.0, "end_sec": 0.3},
            {"start_sec": 0.6, "end_sec": 0.9},
        ],
        boundary,
    ) == [
        {"start_sec": 0.1, "end_sec": 0.3},
        {"start_sec": 0.6, "end_sec": 0.8},
    ]


def test_loss_is_normalized_over_matches_and_upweights_sparse_positives() -> None:
    single_outputs = {
        "eventness_logits": torch.zeros(1, 4),
        "type_logits": torch.zeros(1, 4, 4),
        "activity_logits": torch.zeros(1, 4, 10),
        "onset": torch.zeros(1, 4),
        "offset": torch.ones(1, 4),
        "n_frames": 10,
    }
    target = _events_to_targets(
        [{"type": "speech", "onset": 0.1, "offset": 0.4}],
        n_frames=10,
        max_slots=4,
    )
    single = set_prediction_loss(single_outputs, [target])
    doubled_outputs = {
        key: value.repeat(2, *([1] * (value.ndim - 1)))
        if isinstance(value, torch.Tensor)
        else value
        for key, value in single_outputs.items()
    }
    doubled = set_prediction_loss(doubled_outputs, [target, target])

    assert float(single["eventness_loss"]) > math.log(2.0)
    assert float(doubled["eventness_loss"]) == pytest.approx(
        float(single["eventness_loss"])
    )
    assert float(doubled["type_loss"]) == pytest.approx(float(single["type_loss"]))
    assert float(doubled["activity_loss"]) == pytest.approx(
        float(single["activity_loss"])
    )
    assert float(doubled["boundary_loss"]) == pytest.approx(
        float(single["boundary_loss"])
    )
    assert 0.0 <= float(single["boundary_loss"]) <= 1.0
    assert doubled["n_matched"] == 2.0


def test_decoder_emits_100ms_activity_shape() -> None:
    model = EventSlotDecoder(
        feature_dim=8,
        hidden_dim=16,
        n_slots=3,
        n_heads=4,
        n_layers=1,
        max_duration_sec=2.0,
    )
    output = model(torch.randn(2, 10, 8))
    assert output["eventness_logits"].shape == (2, 3)
    assert output["type_logits"].shape == (2, 3, 4)
    assert output["activity_logits"].shape == (2, 3, 8)
    assert output["onset"].shape == (2, 3)
    assert output["offset"].shape == (2, 3)
    assert torch.all(output["offset"] >= output["onset"])
    assert output["n_frames"] == 8
