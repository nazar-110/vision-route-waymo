"""Video helper functions."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.utils.io import ensure_dir


def ffmpeg_available() -> bool:
    """Return whether ffmpeg is discoverable on PATH."""
    return shutil.which("ffmpeg") is not None


def write_mp4(frames: list[np.ndarray], output_path: str | Path, fps: int = 10) -> None:
    """Write BGR or RGB uint8 frames to an MP4 file."""
    if not frames:
        raise ValueError("Cannot write an empty video")
    path = Path(output_path)
    ensure_dir(path.parent)
    first = frames[0]
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"Expected color frames shaped [H,W,3], got {first.shape}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for {path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("All frames must have the same size")
            writer.write(frame)
    finally:
        writer.release()
