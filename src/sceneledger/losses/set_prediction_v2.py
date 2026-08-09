"""Compatibility boundary-only loss for historical S1a-v2 imports.

The primary experiment uses :mod:`sceneledger.losses.set_prediction`, which
jointly trains activity and boundary heads. This module keeps old imports
working while applying the corrected positive weighting and match
normalization. Historical standalone checkpoints are not architecture-compatible
with the unified dual-head runner and must be retrained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

EVENT_TYPE_TO_INDEX = {"speech": 0, "lys": 1, "music": 2, "sfx": 3}


def _events_to_targets_v2(
    events: list[dict], max_slots: int
) -> dict[str, torch.Tensor | int]:
    type_targets = torch.zeros(max_slots, dtype=torch.long)
    boundary_targets = torch.zeros(max_slots, 2)
    valid = torch.zeros(max_slots, dtype=torch.bool)
    n_events = 0
    for event in events:
        if n_events >= max_slots or event.get("type") not in EVENT_TYPE_TO_INDEX:
            continue
        type_targets[n_events] = EVENT_TYPE_TO_INDEX[event["type"]]
        boundary_targets[n_events] = torch.tensor(
            [float(event["onset"]), float(event["offset"])]
        )
        valid[n_events] = True
        n_events += 1
    return {
        "type_targets": type_targets,
        "boundary_targets": boundary_targets,
        "valid_mask": valid,
        "n_events": n_events,
    }


def _match(
    type_logits: torch.Tensor,
    onset: torch.Tensor,
    offset: torch.Tensor,
    type_targets: torch.Tensor,
    boundary_targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[tuple[int, int]]:
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    if len(valid_indices) == 0:
        return []
    selected_types = type_targets[valid_indices]
    type_cost = -torch.log(
        type_logits.softmax(-1)[:, selected_types].clamp_min(1e-8)
    )
    predicted = torch.stack([onset, offset], dim=-1)[:, None, :]
    expected = boundary_targets[valid_indices][None, :, :]
    normalizer = max(0.1, float(expected.max().item()))
    boundary_cost = (predicted - expected).abs().mean(dim=-1) / normalizer
    rows, columns = linear_sum_assignment(
        (type_cost + boundary_cost).detach().cpu().numpy()
    )
    return [
        (int(row), int(valid_indices[column]))
        for row, column in zip(rows, columns, strict=True)
    ]


def set_prediction_loss_v2(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor | int]],
    boundary_weight: float = 5.0,
) -> dict[str, torch.Tensor]:
    eventness_logits = outputs["eventness_logits"]
    type_logits = outputs["type_logits"]
    onsets = outputs["onset"]
    offsets = outputs["offset"]
    _, n_slots = eventness_logits.shape
    eventness_losses = []
    type_losses = []
    boundary_losses = []

    for batch_index, target in enumerate(targets):
        device = eventness_logits.device
        type_target = target["type_targets"]
        boundary_target = target["boundary_targets"]
        valid_mask = target["valid_mask"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (type_target, boundary_target, valid_mask)
        ):
            raise TypeError("legacy S1a-v2 targets must be tensors")
        type_target = type_target.to(device)
        boundary_target = boundary_target.to(device)
        valid_mask = valid_mask.to(device)
        matched = _match(
            type_logits[batch_index],
            onsets[batch_index],
            offsets[batch_index],
            type_target,
            boundary_target,
            valid_mask,
        )

        eventness_target = torch.zeros(n_slots, device=device)
        for prediction_index, _ in matched:
            eventness_target[prediction_index] = 1.0
        n_positive = len(matched)
        positive_weight = (
            max(1.0, (n_slots - n_positive) / n_positive)
            if n_positive
            else 1.0
        )
        eventness_losses.append(
            F.binary_cross_entropy_with_logits(
                eventness_logits[batch_index],
                eventness_target,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
        )
        for prediction_index, target_index in matched:
            type_losses.append(
                F.cross_entropy(
                    type_logits[batch_index, prediction_index].unsqueeze(0),
                    type_target[target_index].unsqueeze(0),
                )
            )
            predicted_boundary = torch.stack(
                [
                    onsets[batch_index, prediction_index],
                    offsets[batch_index, prediction_index],
                ]
            )
            boundary_losses.append(
                F.l1_loss(predicted_boundary, boundary_target[target_index])
            )

    zero = eventness_logits.sum() * 0.0
    eventness_loss = torch.stack(eventness_losses).mean()
    type_loss = torch.stack(type_losses).mean() if type_losses else zero
    boundary_loss = torch.stack(boundary_losses).mean() if boundary_losses else zero
    return {
        "loss": eventness_loss + type_loss + boundary_weight * boundary_loss,
        "eventness_loss": eventness_loss,
        "type_loss": type_loss,
        "boundary_loss": boundary_loss,
    }


__all__ = ["_events_to_targets_v2", "set_prediction_loss_v2"]
