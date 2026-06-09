"""Camera calibration abstractions for Waymo data and projection tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraCalibration:
    """Minimal pinhole calibration.

    Vehicle coordinates use x forward, y left, z up. For the test camera,
    projection maps vehicle coordinates into an OpenCV-style camera frame:
    camera x right, camera y down, camera z forward.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_xyz_vehicle: tuple[float, float, float] = (0.0, 0.0, 1.35)
    t_camera_vehicle: np.ndarray | None = None
    name: str = "FRONT"
    projection_model: str = "opencv_z_forward"

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


def default_test_calibration(width: int = 320, height: int = 192) -> CameraCalibration:
    """Return a front-camera calibration that makes ground routes visible."""
    return CameraCalibration(
        width=width,
        height=height,
        fx=0.72 * width,
        fy=0.72 * width,
        cx=width / 2.0,
        cy=0.39 * height,
        camera_xyz_vehicle=(0.0, 0.0, 1.35),
        t_camera_vehicle=None,
        name="FRONT",
        projection_model="opencv_z_forward",
    )


def calibration_from_waymo(camera_calibration: Any, width: int, height: int) -> CameraCalibration:
    """Build calibration from Waymo proto fields when the package is installed.

    Waymo camera calibration protos expose an intrinsic vector and an extrinsic
    transform. This function reads those fields dynamically so it can fail
    with a useful message if the installed wheel changes.
    """
    intrinsic = list(getattr(camera_calibration, "intrinsic", []))
    if len(intrinsic) < 4:
        raise ValueError("Waymo camera calibration is missing fx/fy/cx/cy intrinsics")
    fx, fy, cx, cy = map(float, intrinsic[:4])
    transform = getattr(getattr(camera_calibration, "extrinsic", None), "transform", None)
    t_camera_vehicle = None
    if transform is not None:
        values = np.asarray(list(transform), dtype=np.float32)
        if values.size == 16:
            # Waymo sensor extrinsics are conventionally sensor-to-vehicle.
            # We store vehicle-to-camera for projection.
            t_vehicle_camera = values.reshape(4, 4)
            try:
                t_camera_vehicle = np.linalg.inv(t_vehicle_camera).astype(np.float32)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Waymo camera extrinsic matrix is singular") from exc
    name = str(getattr(camera_calibration, "name", "FRONT"))
    return CameraCalibration(
        width=int(width),
        height=int(height),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        t_camera_vehicle=t_camera_vehicle,
        name=name,
        projection_model="waymo_x_forward",
    )
