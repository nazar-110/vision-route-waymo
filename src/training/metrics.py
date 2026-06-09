"""Trajectory metrics."""

from __future__ import annotations

import torch


def displacement_errors(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-waypoint Euclidean displacement errors."""
    return torch.linalg.norm(pred - target, dim=-1)


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average displacement error."""
    return displacement_errors(pred, target).mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Final displacement error."""
    return displacement_errors(pred, target)[:, -1].mean()


def smoothness_score(traj: torch.Tensor) -> torch.Tensor:
    """Mean second-difference magnitude. Lower is smoother."""
    if traj.shape[1] < 3:
        return traj.new_tensor(0.0)
    dd = traj[:, 2:] - 2.0 * traj[:, 1:-1] + traj[:, :-2]
    return torch.linalg.norm(dd, dim=-1).mean()


def min_ade_fde(pred_modes: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute minADE/minFDE for ``[B,K,T,2]`` predictions."""
    if pred_modes.ndim != 4:
        raise ValueError("pred_modes must be shaped [B,K,T,2]")
    errors = torch.linalg.norm(pred_modes - target[:, None], dim=-1)
    mode_ade = errors.mean(dim=-1)
    best = mode_ade.argmin(dim=1)
    batch = torch.arange(target.shape[0], device=target.device)
    best_errors = errors[batch, best]
    return best_errors.mean(), best_errors[:, -1].mean()


def metric_dict(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return common scalar trajectory metrics."""
    return {
        "ADE": float(ade(pred, target).detach().cpu()),
        "FDE": float(fde(pred, target).detach().cpu()),
        "Smoothness": float(smoothness_score(pred).detach().cpu()),
    }
