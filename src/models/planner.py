"""Camera-conditioned ego trajectory planner."""

from __future__ import annotations

import torch
from torch import nn

from src.models.baselines import constant_curvature_baseline, constant_velocity_baseline
from src.models.ego_encoder import EgoMotionEncoder
from src.models.image_encoder import ImageEncoder
from src.models.lidar_encoder import LidarBEVEncoder


class VisionRoutePlanner(nn.Module):
    """Predict future ego waypoints from camera and ego-motion inputs."""

    def __init__(
        self,
        future_steps: int = 20,
        past_steps: int = 16,
        image_backbone: str = "tiny",
        image_feature_dim: int = 128,
        lidar_feature_dim: int = 128,
        ego_feature_dim: int = 64,
        hidden_dim: int = 256,
        command_vocab: int = 4,
        pretrained: bool = False,
        dt: float = 0.25,
        base_strategy: str = "constant_velocity",
        zero_init_residual: bool = False,
        use_lidar: bool = False,
        lidar_channels: int = 3,
    ) -> None:
        super().__init__()
        self.future_steps = int(future_steps)
        self.dt = float(dt)
        self.base_strategy = str(base_strategy)
        self.use_lidar = bool(use_lidar)
        self.image_encoder = ImageEncoder(image_backbone, image_feature_dim, pretrained=pretrained)
        self.lidar_encoder = LidarBEVEncoder(lidar_channels, lidar_feature_dim) if self.use_lidar else None
        self.ego_encoder = EgoMotionEncoder(past_steps, out_dim=ego_feature_dim, command_vocab=command_vocab)
        fusion_dim = image_feature_dim + ego_feature_dim + (lidar_feature_dim if self.use_lidar else 0)
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.future_steps * 2),
        )
        if zero_init_residual:
            final = self.decoder[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(
        self,
        image: torch.Tensor,
        past: torch.Tensor,
        motion: torch.Tensor,
        command: torch.Tensor,
        lidar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        ego_feat = self.ego_encoder(past, motion, command)
        features = [image_feat, ego_feat]
        if self.use_lidar:
            if lidar is None:
                raise ValueError("This model was configured with use_lidar=true but no lidar tensor was provided")
            if self.lidar_encoder is None:
                raise RuntimeError("LiDAR encoder was not initialized")
            features.append(self.lidar_encoder(lidar))
        residual = self.decoder(torch.cat(features, dim=1)).view(image.shape[0], self.future_steps, 2)
        if self.base_strategy == "constant_curvature":
            base = constant_curvature_baseline(past.to(image.dtype), self.future_steps, self.dt)
        elif self.base_strategy == "constant_velocity":
            base = constant_velocity_baseline(past.to(image.dtype), self.future_steps, self.dt)
        else:
            raise ValueError(f"Unknown base_strategy: {self.base_strategy}")
        traj = base + residual
        traj_x = torch.cummax(torch.clamp(traj[..., 0], min=0.1), dim=1).values
        return torch.stack([traj_x, traj[..., 1]], dim=-1)


def build_model_from_config(cfg: dict) -> VisionRoutePlanner:
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    return VisionRoutePlanner(
        future_steps=int(data.get("future_steps", 20)),
        past_steps=int(data.get("past_steps", 16)),
        image_backbone=str(model.get("image_backbone", "tiny")),
        image_feature_dim=int(model.get("image_feature_dim", 128)),
        lidar_feature_dim=int(model.get("lidar_feature_dim", 128)),
        ego_feature_dim=int(model.get("ego_feature_dim", 64)),
        hidden_dim=int(model.get("hidden_dim", 256)),
        command_vocab=int(model.get("command_vocab", 4)),
        pretrained=bool(model.get("pretrained", False)),
        dt=float(data.get("dt", 0.25)),
        base_strategy=str(model.get("base_strategy", "constant_velocity")),
        zero_init_residual=bool(model.get("zero_init_residual", False)),
        use_lidar=bool(model.get("use_lidar", False)),
        lidar_channels=int(model.get("lidar_channels", 3)),
    )
