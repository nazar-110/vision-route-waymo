"""Geometry helpers for trajectory prediction and projection."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def as_points_array(points: np.ndarray | Iterable[Iterable[float]], dims: int = 3) -> np.ndarray:
    """Convert trajectory points to a finite ``[N, dims]`` float32 array."""
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"Expected points shaped [N,2] or [N,3], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Points contain NaN or infinite values")
    if dims == 3 and arr.shape[1] == 2:
        zeros = np.zeros((arr.shape[0], 1), dtype=np.float32)
        arr = np.concatenate([arr, zeros], axis=1)
    if dims == 2 and arr.shape[1] == 3:
        arr = arr[:, :2]
    return arr.astype(np.float32, copy=False)


def make_homogeneous(points: np.ndarray) -> np.ndarray:
    """Append a homogeneous coordinate to ``[N,D]`` points."""
    pts = np.asarray(points, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    return np.concatenate([pts, ones], axis=1)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to 3D points."""
    pts = as_points_array(points, dims=3)
    mat = np.asarray(transform, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4,4), got {mat.shape}")
    out = make_homogeneous(pts) @ mat.T
    return out[:, :3]


def smooth_polyline(points: np.ndarray, weight: float = 0.35, iterations: int = 2) -> np.ndarray:
    """Smooth a 2D polyline while preserving endpoints."""
    pts = np.asarray(points, dtype=np.float32).copy()
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        prev = pts.copy()
        pts[1:-1] = (1.0 - weight) * prev[1:-1] + 0.5 * weight * (prev[:-2] + prev[2:])
    return pts


def trajectory_headings(points: np.ndarray) -> np.ndarray:
    """Return per-segment heading angles in radians for an ``[N,2]`` path."""
    pts = as_points_array(points, dims=2)
    diffs = np.diff(pts, axis=0)
    return np.arctan2(diffs[:, 1], np.maximum(diffs[:, 0], 1e-6))


def curvature_proxy(points: np.ndarray) -> float:
    """A compact smoothness/curvature proxy based on second differences."""
    pts = as_points_array(points, dims=2)
    if len(pts) < 3:
        return 0.0
    dd = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]
    return float(np.mean(np.linalg.norm(dd, axis=-1)))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
