"""Image encoder backbones."""

from __future__ import annotations

import torch
from torch import nn


class TinyCNNEncoder(nn.Module):
    """Fast camera encoder for CPU-friendly Waymo experiments."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class ImageEncoder(nn.Module):
    """Select between a tiny CNN and ResNet18."""

    def __init__(self, backbone: str = "tiny", out_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        backbone = backbone.lower()
        if backbone == "resnet18":
            try:
                from torchvision.models import ResNet18_Weights, resnet18
            except Exception as exc:
                raise RuntimeError("torchvision is required for the ResNet18 backbone") from exc
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, out_dim)
            self.encoder = model
        elif backbone == "tiny":
            self.encoder = TinyCNNEncoder(out_dim)
        else:
            raise ValueError(f"Unsupported image backbone: {backbone}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)
