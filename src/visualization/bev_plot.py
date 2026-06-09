"""Bird's-eye-view trajectory visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.geometry import as_points_array
from src.utils.io import ensure_dir


def save_bev_comparison(
    output_path: str | Path,
    history: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    gt: np.ndarray | None = None,
    title: str = "VisionRoute BEV Comparison",
) -> None:
    """Save a BEV plot comparing predicted and ground-truth future routes."""
    path = Path(output_path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=140)
    ax.set_facecolor("#f7f8fa")
    ax.axhline(0, color="#d0d4d8", linewidth=1)
    ax.axvline(0, color="#d0d4d8", linewidth=1)
    if history is not None:
        h = as_points_array(history, dims=2)
        ax.plot(h[:, 1], h[:, 0], color="#cc8b00", linewidth=2.0, label="Ego history")
        ax.scatter(h[-1:, 1], h[-1:, 0], color="#111111", s=22, zorder=3, label="Current pose")
    if gt is not None:
        g = as_points_array(gt, dims=2)
        ax.plot(g[:, 1], g[:, 0], color="#2da44e", linewidth=3.0, label="Ground truth")
    if pred is not None:
        p = as_points_array(pred, dims=2)
        ax.plot(p[:, 1], p[:, 0], color="#00a2c7", linewidth=3.0, label="Prediction")
        ax.scatter(p[::4, 1], p[::4, 0], color="#00a2c7", s=18)
    ax.set_xlabel("lateral y (m, left positive)")
    ax.set_ylabel("forward x (m)")
    ax.set_title(title)
    ax.grid(True, color="#dfe3e6", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-8, 60)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
