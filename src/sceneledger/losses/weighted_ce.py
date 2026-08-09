"""Time-weighted cross-entropy loss for TAC-style training.

B1 (static SFT) uses ordinary token CE (``timestamp_weight=1.0``).
B2 (TAC paper-spec) upweights the atomic timestamp tokens
``<|t_000|>``..``<|t_300|>`` by ``timestamp_weight`` (default 5.0 per
``configs/experiment_matrix.yaml``) so the model prioritises temporal
precision.

The loss is computed manually (rather than relying on the model's built-in
``CrossEntropyLoss``) so that per-token weights can be applied. ``-100`` labels
are ignored, matching the model's ``ignore_index``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sceneledger.models.tokenizer_utils import compute_timestamp_token_ids


def time_weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    timestamp_token_ids: set[int] | None,
    timestamp_weight: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Per-token CE with optional upweighting of timestamp tokens.

    Args:
        logits: ``[B, T, V]`` model output logits.
        labels: ``[B, T]`` target token IDs (``ignore_index`` masked).
        timestamp_token_ids: token IDs that should be upweighted. If ``None``
            or ``timestamp_weight == 1.0``, this reduces to ordinary CE.
        timestamp_weight: multiplier for timestamp-token positions.
        ignore_index: label value to skip (default -100).

    Returns:
        Scalar mean loss.
    """
    # shift: predict token t+1 from token t (standard causal LM)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # per-token CE (no reduction)
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    # mask valid (non-ignored) positions
    valid_mask = flat_labels != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # compute per-token loss only on valid positions
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    per_token_loss = F.cross_entropy(valid_logits, valid_labels, reduction="none")

    if timestamp_weight == 1.0 or not timestamp_token_ids:
        return per_token_loss.mean()

    # build weight vector: timestamp_weight for timestamp tokens, 1.0 otherwise
    ts_set = torch.tensor(
        sorted(timestamp_token_ids), device=logits.device, dtype=torch.long
    )
    is_timestamp = torch.isin(valid_labels, ts_set)
    weights = torch.where(
        is_timestamp,
        torch.full_like(per_token_loss, timestamp_weight),
        torch.ones_like(per_token_loss),
    )

    return (per_token_loss * weights).sum() / weights.sum()


__all__ = ["compute_timestamp_token_ids", "time_weighted_ce_loss"]
