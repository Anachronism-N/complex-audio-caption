"""Set prediction loss for S1 event slots.

Hungarian matching between predicted slots and target events, followed by
per-slot losses: eventness BCE, type CE, activity Dice. The matching cost
combines type agreement and activity IoU so that the assignment is
permutation-invariant.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from sceneledger.data.schema import TIME_RESOLUTION_SEC
from sceneledger.models.event_slots import N_EVENT_TYPES


def _activity_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two binary activity masks [T]."""
    inter = (a & b).sum().float()
    union = (a | b).sum().float()
    return inter / (union + 1e-6)


def _events_to_targets(
    events: list[dict], n_frames: int, max_slots: int
) -> dict[str, torch.Tensor]:
    """Convert target events (from ledger) to slot-format tensors.

    Returns padded tensors of shape [max_slots, ...] with a validity mask.
    """
    n_events = min(len(events), max_slots)
    type_targets = torch.zeros(max_slots, dtype=torch.long)
    activity_targets = torch.zeros(max_slots, n_frames)
    valid = torch.zeros(max_slots, dtype=torch.bool)

    for i in range(n_events):
        ev = events[i]
        type_idx = {"speech": 0, "lys": 1, "music": 2, "sfx": 3}.get(ev["type"], 0)
        type_targets[i] = type_idx
        # build activity mask from onset/offset
        onset_frame = int(round(ev["onset"] / TIME_RESOLUTION_SEC))
        offset_frame = int(round(ev["offset"] / TIME_RESOLUTION_SEC))
        onset_frame = max(0, min(n_frames, onset_frame))
        offset_frame = max(onset_frame + 1, min(n_frames, offset_frame))
        activity_targets[i, onset_frame:offset_frame] = 1.0
        valid[i] = True

    return {
        "type_targets": type_targets,
        "activity_targets": activity_targets,
        "valid_mask": valid,
        "n_events": n_events,
    }


def hungarian_match(
    type_logits: torch.Tensor,  # [K, 4]
    activity_logits: torch.Tensor,  # [K, T]
    type_targets: torch.Tensor,  # [K_target,]
    activity_targets: torch.Tensor,  # [K_target, T]
    valid_mask: torch.Tensor,  # [K_target,]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predicted slots to target events.

    Returns (matched_pairs, unmatched_pred, unmatched_target) where
    matched_pairs is [(pred_idx, target_idx), ...].
    """
    K_pred = type_logits.shape[0]
    K_target = valid_mask.sum().item()

    if K_target == 0:
        return [], list(range(K_pred)), []

    type_probs = type_logits.softmax(-1)  # [K, 4]
    act_probs = activity_logits.sigmoid()  # [K, T]

    # cost matrix [K_pred, K_target]
    cost = torch.zeros(K_pred, K_target)
    for i in range(K_pred):
        for j in range(K_target):
            if not valid_mask[j]:
                continue
            # type cost: 1 - prob of correct type
            type_cost = 1.0 - type_probs[i, type_targets[j]].item()
            # activity cost: 1 - IoU
            act_pred = act_probs[i] > 0.5
            act_tgt = activity_targets[j] > 0.5
            iou = _activity_iou(act_pred, act_tgt).item()
            cost[i, j] = type_cost + (1.0 - iou)

    # Hungarian
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    # col_ind maps to valid target indices
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    matched = [(int(row_ind[k]), int(valid_indices[col_ind[k]])) for k in range(len(row_ind))]

    matched_pred = {r for r, _ in matched}
    matched_tgt = {t for _, t in matched}
    unmatched_pred = [i for i in range(K_pred) if i not in matched_pred]
    unmatched_tgt = [int(valid_indices[j]) for j in range(K_target) if int(valid_indices[j]) not in matched_tgt]

    return matched, unmatched_pred, unmatched_tgt


def set_prediction_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Compute the S1 set prediction loss for a batch.

    Args:
        outputs: dict from EventSlotDecoder.forward with keys
            eventness_logits [B, K], type_logits [B, K, 4],
            activity_logits [B, K, T].
        targets: list of B target dicts (from _events_to_targets).

    Returns:
        dict with 'loss' (scalar) and component losses.
    """
    eventness_logits = outputs["eventness_logits"]  # [B, K]
    type_logits = outputs["type_logits"]  # [B, K, 4]
    activity_logits = outputs["activity_logits"]  # [B, K, T]
    B, K = eventness_logits.shape

    total_loss = torch.tensor(0.0, device=eventness_logits.device)
    total_eventness = torch.tensor(0.0, device=eventness_logits.device)
    total_type = torch.tensor(0.0, device=eventness_logits.device)
    total_activity = torch.tensor(0.0, device=eventness_logits.device)
    n_matched = 0

    for b in range(B):
        tgt = targets[b]
        # move targets to prediction device
        device = eventness_logits.device
        type_targets = tgt["type_targets"].to(device)
        activity_targets = tgt["activity_targets"].to(device)
        valid_mask = tgt["valid_mask"].to(device)
        n_events = tgt["n_events"]

        # Hungarian matching
        matched, unmatched_pred, unmatched_tgt = hungarian_match(
            type_logits[b], activity_logits[b],
            type_targets, activity_targets, valid_mask,
        )

        # eventness loss: matched slots should be active, unmatched should be null.
        # Use pos_weight to counter class imbalance (most slots are null).
        eventness_target = torch.zeros(K, device=device)
        for pi, _ in matched:
            eventness_target[pi] = 1.0
        pos_weight = torch.tensor([max(1.0, len(matched) / max(1, K - len(matched)) * 3.0)], device=device)
        total_eventness += F.binary_cross_entropy_with_logits(
            eventness_logits[b], eventness_target, reduction="mean", pos_weight=pos_weight
        )

        # type + activity loss on matched slots
        for pi, ti in matched:
            total_type += F.cross_entropy(
                type_logits[b, pi:pi+1], torch.tensor([type_targets[ti]], device=device)
            )
            act_pred = activity_logits[b, pi]  # [T]
            act_tgt = activity_targets[ti]  # [T] already on device
            # Dice loss
            act_p = act_pred.sigmoid()
            inter = (act_p * act_tgt).sum()
            union = act_p.sum() + act_tgt.sum()
            total_activity += 1.0 - 2.0 * inter / (union + 1e-6)
            n_matched += 1

    total_loss = total_eventness + total_type + total_activity
    return {
        "loss": total_loss / max(1, B),
        "eventness_loss": total_eventness / max(1, B),
        "type_loss": total_type / max(1, B),
        "activity_loss": total_activity / max(1, B),
        "n_matched": float(n_matched),
    }


__all__ = ["hungarian_match", "set_prediction_loss", "_events_to_targets"]
