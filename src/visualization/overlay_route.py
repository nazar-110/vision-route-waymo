"""Route overlay rendering on camera images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.calibration import CameraCalibration
from src.utils.geometry import as_points_array, smooth_polyline
from src.utils.io import ensure_dir
from src.visualization.project_points import project_vehicle_points


# OpenCV drawing uses BGR, so this renders as cyan after converting back to RGB.
PRED_COLOR = (255, 210, 34)
GT_COLOR = (60, 235, 95)
HISTORY_COLOR = (245, 205, 80)


def _draw_label(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_polyline(
    canvas: np.ndarray,
    points_vehicle: np.ndarray,
    calibration: CameraCalibration,
    color: tuple[int, int, int],
    thickness: int,
    label: str,
    require_visible: bool = False,
) -> float:
    points = as_points_array(points_vehicle, dims=3)
    pixels, visible, _ = project_vehicle_points(points, calibration, clip=False)
    off_image_pct = 100.0 * float(1.0 - visible.mean()) if len(visible) else 100.0
    visible_pixels = pixels[visible]
    if require_visible and len(visible_pixels) < 2:
        raise ValueError(f"{label} route projection is empty or off-image")
    if len(visible_pixels) >= 2:
        smoothed = smooth_polyline(visible_pixels[:, :2], iterations=3)
        pts = smoothed.round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        for idx, pt in enumerate(smoothed[::4]):
            radius = max(3, thickness // 3)
            cv2.circle(canvas, tuple(pt.round().astype(int)), radius, color, -1, lineType=cv2.LINE_AA)
    return off_image_pct


def draw_route_overlay(
    image_rgb: np.ndarray,
    calibration: CameraCalibration,
    pred: np.ndarray | None = None,
    gt: np.ndarray | None = None,
    history: np.ndarray | None = None,
    frame_text: str | None = None,
    metric_text: str | None = None,
    thickness: int = 8,
    alpha: float = 0.78,
    require_visible: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Draw predicted and ground-truth routes on an RGB image."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")
    base_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = base_bgr.copy()
    diagnostics: dict[str, float] = {}

    if history is not None:
        hist3 = as_points_array(history, dims=3)
        hist3[:, 0] = np.maximum(hist3[:, 0], 0.5)
        diagnostics["history_off_image_pct"] = _draw_polyline(
            overlay, hist3, calibration, HISTORY_COLOR, max(2, thickness // 2), "history", False
        )
    if gt is not None:
        diagnostics["gt_off_image_pct"] = _draw_polyline(overlay, gt, calibration, GT_COLOR, thickness, "gt", False)
    if pred is not None:
        diagnostics["pred_off_image_pct"] = _draw_polyline(
            overlay, pred, calibration, PRED_COLOR, thickness, "prediction", require_visible
        )

    blended = cv2.addWeighted(overlay, alpha, base_bgr, 1.0 - alpha, 0.0)
    legend_bottom = 32
    if gt is not None:
        legend_bottom = 54
    if metric_text:
        legend_bottom = 72
    cv2.rectangle(blended, (8, 8), (182, legend_bottom), (18, 18, 18), -1)
    cv2.circle(blended, (22, 24), 5, PRED_COLOR, -1)
    _draw_label(blended, "Prediction", (34, 29), PRED_COLOR)
    if gt is not None:
        cv2.circle(blended, (22, 46), 5, GT_COLOR, -1)
        _draw_label(blended, "Ground truth", (34, 51), GT_COLOR)
    if frame_text:
        _draw_label(blended, frame_text, (image_rgb.shape[1] - 120, 24), (255, 255, 255))
    if metric_text:
        _draw_label(blended, metric_text, (12, 70), (255, 255, 255))
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), diagnostics


def save_route_overlay(
    image_rgb: np.ndarray,
    calibration: CameraCalibration,
    output_path: str | Path,
    pred: np.ndarray | None = None,
    gt: np.ndarray | None = None,
    history: np.ndarray | None = None,
    frame_text: str | None = None,
    metric_text: str | None = None,
    thickness: int = 8,
    require_visible: bool = True,
) -> dict[str, float]:
    """Draw and save a route overlay PNG."""
    rendered, diagnostics = draw_route_overlay(
        image_rgb,
        calibration,
        pred=pred,
        gt=gt,
        history=history,
        frame_text=frame_text,
        metric_text=metric_text,
        thickness=thickness,
        require_visible=require_visible,
    )
    path = Path(output_path)
    ensure_dir(path.parent)
    bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write overlay image: {path}")
    return diagnostics
