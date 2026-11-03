"""Losses for the two target kinds."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced pixel focal loss (CenterNet), normalised by the number of dots.

    Positives are pixels where the target is 1 (dot centres); negatives near a
    centre are down-weighted by ``(1 - target) ** beta``.
    """
    pos = target.ge(0.999).float()
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    p = log_p.exp()
    pos_loss = -(log_p * (1 - p) ** alpha * pos).sum()
    neg_loss = -(log_1mp * p**alpha * (1 - target) ** beta * (1 - pos)).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1.0)


def mask_bce_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 5.0) -> torch.Tensor:
    """Weighted BCE on binary dot masks, as in the original project (pos_weight 5)."""
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))


LOSSES = {"heatmap": focal_heatmap_loss, "mask": mask_bce_loss}
