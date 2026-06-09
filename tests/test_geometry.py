from __future__ import annotations

import numpy as np

from src.utils.geometry import curvature_proxy, smooth_polyline, transform_points


def test_transform_points_identity() -> None:
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    out = transform_points(pts, np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(out, pts)


def test_smooth_polyline_preserves_endpoints() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]], dtype=np.float32)
    smoothed = smooth_polyline(pts, iterations=3)
    np.testing.assert_allclose(smoothed[0], pts[0])
    np.testing.assert_allclose(smoothed[-1], pts[-1])
    assert curvature_proxy(smoothed) < curvature_proxy(pts)
