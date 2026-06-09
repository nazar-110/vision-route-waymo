"""Ego-motion and route-command encoders."""

from __future__ import annotations

import torch
from torch import nn


class EgoMotionEncoder(nn.Module):
    """Encode past trajectory, velocity/acceleration features, and command."""

    def __init__(
        self,
        past_steps: int,
        motion_dim: int = 8,
        command_vocab: int = 4,
        command_dim: int = 8,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.command_embedding = nn.Embedding(command_vocab, command_dim)
        in_dim = past_steps * 2 + motion_dim + command_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, past: torch.Tensor, motion: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        command = command.clamp(min=0, max=self.command_embedding.num_embeddings - 1)
        cmd = self.command_embedding(command.long())
        x = torch.cat([past.flatten(start_dim=1), motion.float(), cmd], dim=1)
        return self.net(x)
