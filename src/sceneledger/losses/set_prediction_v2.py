"""Set prediction loss v2: boundary regression (L1) instead of activity Dice."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

N_EVENT_TYPES = 4


def _events_to_targets_v2(events: list[dict], max_slots: int) -> dict[str, torch.Tensor]:
    n_events = min(len(events), max_slots)
    type_targets = torch.zeros(max_slots, dtype=torch.long)
    boundary_targets = torch.zeros(max_slots, 2)  # [onset, offset]
    valid = torch.zeros(max_slots, dtype=torch.bool)
    for i in range(n_events):
        ev = events[i]
        type_idx = {"speech": 0, "lys": 1, "music": 2, "sfx": 3}.get(ev["type"], 0)
        type_targets[i] = type_idx
        boundary_targets[i, 0] = ev["onset"]
        boundary_targets[i, 1] = ev["offset"]
        valid[i] = True
    return {"type_targets": type_targets, "boundary_targets": boundary_targets, "valid_mask": valid, "n_events": n_events}


def _hungarian_match_v2(
    type_logits, onset, offset, type_targets, boundary_targets, valid_mask,
):
    K_pred = type_logits.shape[0]
    K_target = valid_mask.sum().item()
    if K_target == 0:
        return [], list(range(K_pred)), []

    type_probs = type_logits.softmax(-1)
    cost = torch.zeros(K_pred, K_target)
    for i in range(K_pred):
        for j in range(K_target):
            if not valid_mask[j]:
                continue
            type_cost = 1.0 - type_probs[i, type_targets[j]].item()
            # boundary cost: normalized L1
            t_onset = boundary_targets[j, 0].item()
            t_offset = boundary_targets[j, 1].item()
            p_onset = onset[i].item()
            p_offset = offset[i].item()
            dur = max(0.1, t_offset - t_onset)
            bnd_cost = (abs(p_onset - t_onset) + abs(p_offset - t_offset)) / (2 * dur)
            cost[i, j] = type_cost + bnd_cost

    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    matched = [(int(row_ind[k]), int(valid_indices[col_ind[k]])) for k in range(len(row_ind))]
    matched_pred = {r for r, _ in matched}
    matched_tgt = {t for _, t in matched}
    unmatched_pred = [i for i in range(K_pred) if i not in matched_pred]
    unmatched_tgt = [int(valid_indices[j]) for j in range(K_target) if int(valid_indices[j]) not in matched_tgt]
    return matched, unmatched_pred, unmatched_tgt


def set_prediction_loss_v2(outputs, targets, boundary_weight=5.0):
    eventness_logits = outputs["eventness_logits"]
    type_logits = outputs["type_logits"]
    onsets = outputs["onset"]
    offsets = outputs["offset"]
    B, K = eventness_logits.shape
    device = eventness_logits.device

    total_ev = torch.tensor(0.0, device=device)
    total_type = torch.tensor(0.0, device=device)
    total_bnd = torch.tensor(0.0, device=device)

    for b in range(B):
        tgt = targets[b]
        type_targets = tgt["type_targets"].to(device)
        boundary_targets = tgt["boundary_targets"].to(device)
        valid_mask = tgt["valid_mask"].to(device)

        matched, _, _ = _hungarian_match_v2(
            type_logits[b], onsets[b], offsets[b],
            type_targets, boundary_targets, valid_mask,
        )

        # eventness with pos_weight
        ev_target = torch.zeros(K, device=device)
        for pi, _ in matched:
            ev_target[pi] = 1.0
        pos_weight = torch.tensor([max(1.0, len(matched) / max(1, K - len(matched)) * 3.0)], device=device)
        total_ev += F.binary_cross_entropy_with_logits(
            eventness_logits[b], ev_target, reduction="mean", pos_weight=pos_weight
        )

        for pi, ti in matched:
            total_type += F.cross_entropy(
                type_logits[b, pi:pi+1],
                torch.tensor([type_targets[ti]], device=device),
            )
            # L1 boundary loss
            pred_bnd = torch.stack([onsets[b, pi], offsets[b, pi]])
            tgt_bnd = boundary_targets[ti]
            total_bnd += F.l1_loss(pred_bnd, tgt_bnd)

    total_loss = total_ev + total_type + boundary_weight * total_bnd
    return {
        "loss": total_loss / max(1, B),
        "eventness_loss": total_ev / max(1, B),
        "type_loss": total_type / max(1, B),
        "boundary_loss": total_bnd / max(1, B),
    }


__all__ = ["set_prediction_loss_v2", "_events_to_targets_v2"]
