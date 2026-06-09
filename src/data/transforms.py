"""Image and trajectory transforms."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB image."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an RGB uint8 image to a float tensor in CHW format."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor
