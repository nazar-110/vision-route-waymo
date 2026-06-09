"""LiDAR point-cloud helpers for Waymo Perception data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LidarBEVConfig:
    """Bird's-eye-view rasterization settings in vehicle coordinates."""

    x_min: float = 0.0
    x_max: float = 70.0
    y_min: float = -35.0
    y_max: float = 35.0
    z_min: float = -3.0
    z_max: float = 3.0
    height: int = 160
    width: int = 160


def lidar_config_from_dict(cfg: dict[str, Any] | None) -> LidarBEVConfig:
    """Build a LiDAR BEV config from YAML values."""
    cfg = cfg or {}
    return LidarBEVConfig(
        x_min=float(cfg.get("x_min", 0.0)),
        x_max=float(cfg.get("x_max", 70.0)),
        y_min=float(cfg.get("y_min", -35.0)),
        y_max=float(cfg.get("y_max", 35.0)),
        z_min=float(cfg.get("z_min", -3.0)),
        z_max=float(cfg.get("z_max", 3.0)),
        height=int(cfg.get("height", 160)),
        width=int(cfg.get("width", 160)),
    )


def waymo_frame_to_points(frame: Any) -> np.ndarray:
    """Convert Waymo range images into one vehicle-frame point cloud."""
    try:
        from waymo_open_dataset.utils import frame_utils  # type: ignore
    except Exception as exc:
        raise RuntimeError("waymo_open_dataset.utils.frame_utils is required for LiDAR parsing") from exc

    range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(
        frame
    )
    point_sets, _ = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
    )
    if not point_sets:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate([np.asarray(points, dtype=np.float32) for points in point_sets if len(points)], axis=0)


def points_to_bev(points: np.ndarray, cfg: LidarBEVConfig) -> np.ndarray:
    """Rasterize vehicle-frame points into density, max-height, and occupancy channels."""
    bev = np.zeros((3, cfg.height, cfg.width), dtype=np.float32)
    if points.size == 0:
        return bev

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    mask = (
        (x >= cfg.x_min)
        & (x <= cfg.x_max)
        & (y >= cfg.y_min)
        & (y <= cfg.y_max)
        & (z >= cfg.z_min)
        & (z <= cfg.z_max)
    )
    if not np.any(mask):
        return bev

    x = x[mask]
    y = y[mask]
    z = z[mask]
    rows = ((cfg.x_max - x) / (cfg.x_max - cfg.x_min) * (cfg.height - 1)).astype(np.int32)
    cols = ((y - cfg.y_min) / (cfg.y_max - cfg.y_min) * (cfg.width - 1)).astype(np.int32)
    rows = np.clip(rows, 0, cfg.height - 1)
    cols = np.clip(cols, 0, cfg.width - 1)

    counts = np.zeros((cfg.height, cfg.width), dtype=np.float32)
    max_height = np.full((cfg.height, cfg.width), cfg.z_min, dtype=np.float32)
    np.add.at(counts, (rows, cols), 1.0)
    np.maximum.at(max_height, (rows, cols), z)

    bev[0] = np.clip(np.log1p(counts) / np.log(32.0), 0.0, 1.0)
    bev[1] = np.clip((max_height - cfg.z_min) / (cfg.z_max - cfg.z_min), 0.0, 1.0)
    bev[2] = (counts > 0).astype(np.float32)
    return bev


def waymo_frame_to_bev(frame: Any, cfg: LidarBEVConfig) -> np.ndarray:
    """Convert a Waymo frame's LiDAR returns into a compact BEV tensor."""
    return points_to_bev(waymo_frame_to_points(frame), cfg)
