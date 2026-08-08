from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .slots import containment_loss


def slot_set_loss(
    outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]
) -> dict[str, Tensor]:
    """Permutation-invariant structural loss for variable-size track/event targets.

    Target keys: track_type [Nt], track_activity [Nt,T], event_type [Ne],
    event_activity [Ne,T], event_track [Ne] indexing target tracks.
    """
    device = outputs["track_presence_logits"].device
    losses = {
        name: torch.zeros((), device=device)
        for name in (
            "track_presence",
            "track_type",
            "track_activity",
            "eventness",
            "event_type",
            "event_activity",
            "pointer",
            "containment",
        )
    }
    batch_size = len(targets)
    for batch_index, target in enumerate(targets):
        if len(target["track_type"]) > outputs["track_presence_logits"].shape[1]:
            raise ValueError("Target track count exceeds configured track slots")
        if len(target["event_type"]) > outputs["eventness_logits"].shape[1]:
            raise ValueError("Target event count exceeds configured event slots")
        expected_frames = outputs["track_activity_logits"].shape[-1]
        if target["track_activity"].shape[-1] != expected_frames:
            raise ValueError("Track target and model activity grids differ")
        if target["event_activity"].shape[-1] != expected_frames:
            raise ValueError("Event target and model activity grids differ")
        track_pairs = _match_slots(
            outputs["track_type_logits"][batch_index],
            outputs["track_activity_logits"][batch_index],
            target["track_type"].to(device),
            target["track_activity"].to(device),
        )
        track_presence_target = torch.zeros_like(outputs["track_presence_logits"][batch_index])
        if track_pairs:
            predicted_tracks = torch.tensor([pair[0] for pair in track_pairs], device=device)
            target_tracks = torch.tensor([pair[1] for pair in track_pairs], device=device)
            track_presence_target[predicted_tracks] = 1.0
            losses["track_type"] += nn.functional.cross_entropy(
                outputs["track_type_logits"][batch_index, predicted_tracks],
                target["track_type"].to(device)[target_tracks],
            )
            losses["track_activity"] += _activity_loss(
                outputs["track_activity_logits"][batch_index, predicted_tracks],
                target["track_activity"].to(device)[target_tracks],
            )
        losses["track_presence"] += nn.functional.binary_cross_entropy_with_logits(
            outputs["track_presence_logits"][batch_index], track_presence_target
        )

        event_pairs = _match_slots(
            outputs["event_type_logits"][batch_index],
            outputs["event_activity_logits"][batch_index],
            target["event_type"].to(device),
            target["event_activity"].to(device),
        )
        eventness_target = torch.zeros_like(outputs["eventness_logits"][batch_index])
        if event_pairs:
            predicted_events = torch.tensor([pair[0] for pair in event_pairs], device=device)
            target_events = torch.tensor([pair[1] for pair in event_pairs], device=device)
            eventness_target[predicted_events] = 1.0
            losses["event_type"] += nn.functional.cross_entropy(
                outputs["event_type_logits"][batch_index, predicted_events],
                target["event_type"].to(device)[target_events],
            )
            losses["event_activity"] += _activity_loss(
                outputs["event_activity_logits"][batch_index, predicted_events],
                target["event_activity"].to(device)[target_events],
            )
            target_to_predicted_track = {
                target_index: predicted for predicted, target_index in track_pairs
            }
            pointer_targets = torch.tensor(
                [
                    target_to_predicted_track[int(target["event_track"][target_index])]
                    for target_index in target_events.tolist()
                ],
                device=device,
            )
            losses["pointer"] += nn.functional.cross_entropy(
                outputs["track_pointer_logits"][batch_index, predicted_events], pointer_targets
            )
            losses["containment"] += containment_loss(
                outputs["event_activity_logits"][batch_index : batch_index + 1, predicted_events],
                outputs["track_activity_logits"][batch_index : batch_index + 1],
                pointer_targets.unsqueeze(0),
            )
        losses["eventness"] += nn.functional.binary_cross_entropy_with_logits(
            outputs["eventness_logits"][batch_index], eventness_target
        )
    return {name: value / max(batch_size, 1) for name, value in losses.items()}


def _match_slots(
    predicted_type: Tensor,
    predicted_activity: Tensor,
    target_type: Tensor,
    target_activity: Tensor,
) -> list[tuple[int, int]]:
    if target_type.numel() == 0:
        return []
    type_cost = -predicted_type.log_softmax(-1)[:, target_type]
    predicted_probability = predicted_activity.sigmoid()
    intersection = torch.einsum("pt,nt->pn", predicted_probability, target_activity.float())
    denominator = predicted_probability.sum(-1, keepdim=True) + target_activity.float().sum(
        -1
    ).unsqueeze(0)
    dice_cost = 1.0 - (2 * intersection + 1.0) / (denominator + 1.0)
    cost = (0.4 * type_cost + 0.6 * dice_cost).detach().cpu().numpy()
    predicted_indices, target_indices = linear_sum_assignment(np.asarray(cost))
    return list(zip(predicted_indices.tolist(), target_indices.tolist()))


def _activity_loss(logits: Tensor, target: Tensor) -> Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target.float())
    probability = logits.sigmoid()
    dice = 1.0 - (2 * (probability * target).sum(-1) + 1.0) / (
        probability.sum(-1) + target.sum(-1) + 1.0
    )
    return bce + dice.mean()
