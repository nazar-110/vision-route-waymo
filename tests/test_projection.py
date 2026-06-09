from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.calibration import default_test_calibration
from src.visualization.overlay_route import save_route_overlay
from src.visualization.project_points import project_vehicle_points


def test_points_in_front_project_inside_image() -> None:
    cal = default_test_calibration()
    pts = np.array([[8.0, 0.0, 0.0], [18.0, 0.5, 0.0], [32.0, -0.5, 0.0]], dtype=np.float32)
    pixels, visible, depth = project_vehicle_points(pts, cal)
    assert visible.all()
    assert (depth > 0).all()
    assert ((pixels[:, 0] >= 0) & (pixels[:, 0] < cal.width)).all()
    assert ((pixels[:, 1] >= 0) & (pixels[:, 1] < cal.height)).all()


def test_points_behind_camera_are_filtered() -> None:
    cal = default_test_calibration()
    pts = np.array([[-2.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=np.float32)
    _, visible, _ = project_vehicle_points(pts, cal)
    assert not bool(visible[0])
    assert bool(visible[1])


def test_left_right_and_horizon_behavior() -> None:
    cal = default_test_calibration()
    straight = np.array([[20.0, 0.0, 0.0]], dtype=np.float32)
    left = np.array([[20.0, 2.0, 0.0]], dtype=np.float32)
    right = np.array([[20.0, -2.0, 0.0]], dtype=np.float32)
    near = np.array([[8.0, 0.0, 0.0]], dtype=np.float32)
    far = np.array([[35.0, 0.0, 0.0]], dtype=np.float32)
    p_straight, _, _ = project_vehicle_points(straight, cal)
    p_left, _, _ = project_vehicle_points(left, cal)
    p_right, _, _ = project_vehicle_points(right, cal)
    p_near, _, _ = project_vehicle_points(near, cal)
    p_far, _, _ = project_vehicle_points(far, cal)
    assert p_left[0, 0] < p_straight[0, 0] < p_right[0, 0]
    assert p_far[0, 1] < p_near[0, 1]


def test_overlay_is_visible_with_test_calibration(tmp_path: Path) -> None:
    cal = default_test_calibration()
    image = np.zeros((cal.height, cal.width, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)
    route = np.stack([np.linspace(6.0, 34.0, 20), np.linspace(0.0, 1.2, 20)], axis=1).astype(np.float32)
    out = tmp_path / "overlay.png"
    diagnostics = save_route_overlay(image, cal, out, pred=route, gt=route)
    assert out.exists()
    image = cv2.imread(str(out))
    assert image is not None
    assert diagnostics["pred_off_image_pct"] < 80.0
