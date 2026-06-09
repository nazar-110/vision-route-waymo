"""Trajectory losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def trajectory_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "smooth_l1") -> torch.Tensor:
    """Compute a waypoint regression loss."""
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unknown trajectory loss: {loss_type}")


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalize second differences along the future trajectory."""
    if pred.shape[1] < 3:
        return pred.new_tensor(0.0)
    dd = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    return torch.mean(torch.linalg.norm(dd, dim=-1))


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "smooth_l1",
    smoothness_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    reg = trajectory_loss(pred, target, loss_type)
    smooth = smoothness_loss(pred)
    total = reg + float(smoothness_weight) * smooth
    return total, {"regression": float(reg.detach().cpu()), "smoothness": float(smooth.detach().cpu())}
