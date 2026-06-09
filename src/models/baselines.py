"""Non-neural trajectory baselines."""

from __future__ import annotations

import torch


def constant_velocity_baseline(past: torch.Tensor, future_steps: int, dt: float = 0.25) -> torch.Tensor:
    """Extrapolate the latest ego velocity in vehicle coordinates."""
    if past.shape[1] < 2:
        raise ValueError("Need at least two past points for constant velocity baseline")
    velocity = (past[:, -1] - past[:, -2]) / float(dt)
    steps = torch.arange(1, future_steps + 1, device=past.device, dtype=past.dtype).view(1, -1, 1)
    return past[:, -1:].clone() + velocity[:, None, :] * steps * float(dt)


def constant_curvature_baseline(past: torch.Tensor, future_steps: int, dt: float = 0.25) -> torch.Tensor:
    """Roll out a smooth path using recent velocity and lateral acceleration."""
    if past.shape[1] < 4:
        return constant_velocity_baseline(past, future_steps, dt)
    velocity = (past[:, -1] - past[:, -2]) / float(dt)
    prev_velocity = (past[:, -2] - past[:, -3]) / float(dt)
    accel = (velocity - prev_velocity) / float(dt)
    steps = torch.arange(1, future_steps + 1, device=past.device, dtype=past.dtype).view(1, -1, 1)
    t = steps * float(dt)
    pred = past[:, -1:].clone() + velocity[:, None, :] * t + 0.5 * accel[:, None, :] * t * t
    pred_x = torch.cummax(pred[..., 0], dim=1).values
    return torch.stack([pred_x, pred[..., 1]], dim=-1)
