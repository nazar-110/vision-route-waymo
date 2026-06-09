from __future__ import annotations

import torch

from src.models.baselines import constant_curvature_baseline, constant_velocity_baseline
from src.training.metrics import ade, fde, metric_dict, smoothness_score


def test_ade_fde_zero_for_exact_match() -> None:
    target = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    pred = target.clone()
    assert ade(pred, target).item() == 0.0
    assert fde(pred, target).item() == 0.0


def test_smoothness_positive_for_kink() -> None:
    traj = torch.tensor([[[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]])
    assert smoothness_score(traj).item() > 0.0


def test_baselines_shape() -> None:
    past = torch.tensor(
        [
            [
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.1],
                [2.0, 0.2],
            ]
        ]
    )
    cv = constant_velocity_baseline(past, 5)
    cc = constant_curvature_baseline(past, 5)
    assert cv.shape == (1, 5, 2)
    assert cc.shape == (1, 5, 2)
    metrics = metric_dict(cv, cc)
    assert set(metrics) == {"ADE", "FDE", "Smoothness"}
