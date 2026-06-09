"""Projection from vehicle-coordinate trajectories to image pixels."""

from __future__ import annotations

import numpy as np

from src.data.calibration import CameraCalibration
from src.utils.geometry import as_points_array, transform_points


def vehicle_to_camera(points_vehicle: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    """Convert vehicle-frame points to OpenCV camera coordinates."""
    points = as_points_array(points_vehicle, dims=3)
    if calibration.t_camera_vehicle is not None:
        return transform_points(points, calibration.t_camera_vehicle)

    camera_origin = np.asarray(calibration.camera_xyz_vehicle, dtype=np.float32)
    rel = points - camera_origin[None, :]
    x_forward = rel[:, 0]
    y_left = rel[:, 1]
    z_up = rel[:, 2]
    x_right = -y_left
    y_down = -z_up
    z_forward = x_forward
    return np.stack([x_right, y_down, z_forward], axis=1).astype(np.float32)


def project_vehicle_points(
    points_vehicle: np.ndarray,
    calibration: CameraCalibration,
    min_depth: float = 0.25,
    clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project vehicle points into pixels.

    Returns ``pixels, visible_mask, depth`` where ``visible_mask`` includes
    positive depth and image bounds unless ``clip`` is true.
    """
    camera_points = vehicle_to_camera(points_vehicle, calibration)
    if not np.all(np.isfinite(camera_points)):
        raise ValueError("Projection produced non-finite camera coordinates")

    if calibration.projection_model == "waymo_x_forward":
        depth = camera_points[:, 0]
        valid_depth = depth > min_depth
        safe_depth = np.where(valid_depth, depth, 1.0)
        u = calibration.fx * (-camera_points[:, 1] / safe_depth) + calibration.cx
        v = calibration.fy * (-camera_points[:, 2] / safe_depth) + calibration.cy
        pixels = np.stack([u, v], axis=1).astype(np.float32)
    else:
        depth = camera_points[:, 2]
        valid_depth = depth > min_depth
        safe_depth = np.where(valid_depth, depth, 1.0)
        u = calibration.fx * (camera_points[:, 0] / safe_depth) + calibration.cx
        v = calibration.fy * (camera_points[:, 1] / safe_depth) + calibration.cy
        pixels = np.stack([u, v], axis=1).astype(np.float32)

    if not np.all(np.isfinite(pixels)):
        raise ValueError("Projection produced NaN or infinite image coordinates")

    in_bounds = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < calibration.width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < calibration.height)
    )
    visible = valid_depth & in_bounds
    if clip:
        pixels[:, 0] = np.clip(pixels[:, 0], 0, calibration.width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, calibration.height - 1)
        visible = valid_depth
    return pixels, visible, depth.astype(np.float32)
