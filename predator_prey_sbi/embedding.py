from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TimeSeriesEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, hidden_channels, kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding
        )
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_channels, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("Embedding input must have shape (batch, channels, time)")
        in_channels = self.conv1.in_channels
        if x.size(1) != in_channels and x.size(2) == in_channels:
            x = x.transpose(1, 2)
        if x.size(1) != in_channels:
            raise ValueError("Embedding input has unexpected channel dimension")
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


def build_embedding_net(config: dict[str, Any] | None) -> nn.Module:
    cfg = config or {}
    in_channels = int(cfg.get("in_channels", 4))
    hidden_channels = int(cfg.get("hidden_channels", 16))
    embedding_dim = int(cfg.get("embedding_dim", 32))
    kernel_size = int(cfg.get("kernel_size", 5))
    return TimeSeriesEmbedding(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
    )
