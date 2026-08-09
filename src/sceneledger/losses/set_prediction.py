"""Permutation-invariant matching and losses for S1 event slots.

Hungarian matching uses event type probability and a soft activity Dice cost.
After assignment, eventness is learned for every slot while type and activity
losses are averaged over matched events. This keeps scenes with many events
from receiving an accidental larger loss solely because they contain more
matches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from sceneledger.data.schema import TIME_RESOLUTION_SEC

EVENT_TYPE_TO_INDEX = {"speech": 0, "lys": 1, "music": 2, "sfx": 3}


def _event_spans(event: dict) -> list[dict[str, float]]:
    spans = event.get("spans")
    if spans:
        return [
            {
                "start_sec": float(span["start_sec"]),
                "end_sec": float(span["end_sec"]),
            }
            for span in spans
        ]
    if "onset" in event and "offset" in event:
        return [
            {
                "start_sec": float(event["onset"]),
                "end_sec": float(event["offset"]),
            }
        ]
    return []


def _events_to_targets(
    events: list[dict], n_frames: int, max_slots: int
) -> dict[str, torch.Tensor | int]:
    """Convert Ledger-like events to padded slot targets.

    All disjoint spans are preserved on the 100 ms activity grid. Events that
    fall completely outside the represented audio are excluded explicitly.
    """
    type_targets = torch.zeros(max_slots, dtype=torch.long)
    activity_targets = torch.zeros(max_slots, n_frames)
    valid = torch.zeros(max_slots, dtype=torch.bool)

    target_index = 0
    for event in events:
        if target_index >= max_slots:
            break
        event_type = event.get("type")
        if event_type not in EVENT_TYPE_TO_INDEX:
            continue

        mask = torch.zeros(n_frames)
        for span in _event_spans(event):
            start = int(round(span["start_sec"] / TIME_RESOLUTION_SEC))
            end = int(round(span["end_sec"] / TIME_RESOLUTION_SEC))
            start = max(0, min(n_frames, start))
            end = max(0, min(n_frames, end))
            if end <= start and start < n_frames:
                end = start + 1
            if end > start:
                mask[start:end] = 1.0
        if not mask.any():
            continue

        type_targets[target_index] = EVENT_TYPE_TO_INDEX[event_type]
        activity_targets[target_index] = mask
        valid[target_index] = True
        target_index += 1

    return {
        "type_targets": type_targets,
        "activity_targets": activity_targets,
        "valid_mask": valid,
        "n_events": target_index,
    }


def hungarian_match(
    type_logits: torch.Tensor,
    activity_logits: torch.Tensor,
    type_targets: torch.Tensor,
    activity_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    type_cost_weight: float = 1.0,
    activity_cost_weight: float = 2.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predicted slots to valid targets with a soft assignment cost."""
    n_predictions = type_logits.shape[0]
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    if len(valid_indices) == 0:
        return [], list(range(n_predictions)), []

    selected_types = type_targets[valid_indices]
    selected_activity = activity_targets[valid_indices].float()
    type_probability = type_logits.softmax(-1)[:, selected_types]
    type_cost = -torch.log(type_probability.clamp_min(1e-8))

    predicted_activity = activity_logits.sigmoid()[:, None, :]
    target_activity = selected_activity[None, :, :]
    intersection = (predicted_activity * target_activity).sum(dim=-1)
    denominator = predicted_activity.sum(dim=-1) + target_activity.sum(dim=-1)
    activity_cost = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

    cost = type_cost_weight * type_cost + activity_cost_weight * activity_cost
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    matched = [
        (int(row), int(valid_indices[column]))
        for row, column in zip(rows, columns)  # noqa: B905 - Python 3.10 target
    ]

    matched_predictions = {prediction for prediction, _ in matched}
    matched_targets = {target for _, target in matched}
    unmatched_predictions = [
        index for index in range(n_predictions) if index not in matched_predictions
    ]
    unmatched_targets = [
        int(index) for index in valid_indices if int(index) not in matched_targets
    ]
    return matched, unmatched_predictions, unmatched_targets


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def set_prediction_loss(
    outputs: dict[str, torch.Tensor | int],
    targets: list[dict[str, torch.Tensor | int]],
    *,
    eventness_weight: float = 1.0,
    type_weight: float = 1.0,
    activity_weight: float = 2.0,
    positive_weight_scale: float = 1.0,
    max_positive_weight: float = 20.0,
) -> dict[str, torch.Tensor | float]:
    """Compute normalized eventness, type and activity losses for a batch."""
    eventness_logits = outputs["eventness_logits"]
    type_logits = outputs["type_logits"]
    activity_logits = outputs["activity_logits"]
    if not isinstance(eventness_logits, torch.Tensor):  # defensive type narrowing
        raise TypeError("eventness_logits must be a tensor")
    if not isinstance(type_logits, torch.Tensor):
        raise TypeError("type_logits must be a tensor")
    if not isinstance(activity_logits, torch.Tensor):
        raise TypeError("activity_logits must be a tensor")

    batch_size, n_slots = eventness_logits.shape
    if len(targets) != batch_size:
        raise ValueError(f"received {len(targets)} targets for batch size {batch_size}")

    eventness_losses: list[torch.Tensor] = []
    type_losses: list[torch.Tensor] = []
    activity_losses: list[torch.Tensor] = []
    total_matches = 0

    for batch_index, target in enumerate(targets):
        device = eventness_logits.device
        type_target = target["type_targets"]
        activity_target = target["activity_targets"]
        valid_mask = target["valid_mask"]
        if not isinstance(type_target, torch.Tensor):
            raise TypeError("type_targets must be a tensor")
        if not isinstance(activity_target, torch.Tensor):
            raise TypeError("activity_targets must be a tensor")
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a tensor")
        type_target = type_target.to(device)
        activity_target = activity_target.to(device)
        valid_mask = valid_mask.to(device)

        if activity_target.shape[-1] != activity_logits.shape[-1]:
            raise ValueError("target and prediction activity grids differ")
        matched, _, _ = hungarian_match(
            type_logits[batch_index],
            activity_logits[batch_index],
            type_target,
            activity_target,
            valid_mask,
        )

        eventness_target = torch.zeros(n_slots, device=device)
        for prediction_index, _ in matched:
            eventness_target[prediction_index] = 1.0
        n_positive = len(matched)
        if n_positive:
            imbalance = (n_slots - n_positive) / n_positive
            positive_weight = min(
                max_positive_weight, max(1.0, imbalance * positive_weight_scale)
            )
        else:
            positive_weight = 1.0
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
            probability = activity_logits[
                batch_index, prediction_index
            ].sigmoid()
            expected = activity_target[target_index]
            intersection = (probability * expected).sum()
            denominator = probability.sum() + expected.sum()
            activity_losses.append(
                1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            )
        total_matches += n_positive

    eventness_loss = torch.stack(eventness_losses).mean()
    type_loss = (
        torch.stack(type_losses).mean()
        if type_losses
        else _zero_like(type_logits)
    )
    activity_loss = (
        torch.stack(activity_losses).mean()
        if activity_losses
        else _zero_like(activity_logits)
    )
    loss = (
        eventness_weight * eventness_loss
        + type_weight * type_loss
        + activity_weight * activity_loss
    )
    return {
        "loss": loss,
        "eventness_loss": eventness_loss,
        "type_loss": type_loss,
        "activity_loss": activity_loss,
        "n_matched": float(total_matches),
    }


__all__ = ["_events_to_targets", "hungarian_match", "set_prediction_loss"]
