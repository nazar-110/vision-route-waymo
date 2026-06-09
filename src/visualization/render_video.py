"""Video rendering for camera route overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.calibration import CameraCalibration
from src.utils.video import write_mp4
from src.visualization.overlay_route import draw_route_overlay


def render_overlay_video(
    images_rgb: list[np.ndarray],
    calibrations: list[CameraCalibration],
    preds: list[np.ndarray],
    gts: list[np.ndarray] | None,
    output_path: str | Path,
    histories: list[np.ndarray] | None = None,
    fps: int = 10,
    metric_texts: list[str] | None = None,
    thickness: int = 8,
    require_visible: bool = True,
) -> None:
    """Render and save a camera route overlay MP4."""
    if not (len(images_rgb) == len(calibrations) == len(preds)):
        raise ValueError("images, calibrations, and predictions must have matching lengths")
    frames_bgr: list[np.ndarray] = []
    for idx, image in enumerate(images_rgb):
        gt = gts[idx] if gts is not None else None
        history = histories[idx] if histories is not None else None
        metric = metric_texts[idx] if metric_texts is not None else None
        rendered, _ = draw_route_overlay(
            image,
            calibrations[idx],
            pred=preds[idx],
            gt=gt,
            history=history,
            frame_text=f"frame {idx:03d}",
            metric_text=metric,
            thickness=thickness,
            require_visible=require_visible,
        )
        frames_bgr.append(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    write_mp4(frames_bgr, output_path, fps=fps)
