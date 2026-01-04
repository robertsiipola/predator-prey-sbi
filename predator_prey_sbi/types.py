from __future__ import annotations

from typing import Protocol

import torch


class PosteriorLike(Protocol):
    def sample(self, shape: tuple[int, ...], x: torch.Tensor) -> torch.Tensor: ...


class PriorLike(Protocol):
    def sample(self, shape: tuple[int, ...] | torch.Size) -> torch.Tensor: ...
