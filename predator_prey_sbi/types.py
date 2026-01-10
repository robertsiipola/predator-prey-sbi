from __future__ import annotations

from typing import Protocol, Self

import torch


class PosteriorLike(Protocol):
    def sample(
        self,
        shape: tuple[int, ...] | torch.Size,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def log_prob(self, value: torch.Tensor) -> torch.Tensor: ...

    def set_default_x(self, x: torch.Tensor) -> Self: ...


class PriorLike(Protocol):
    def sample(self, shape: tuple[int, ...] | torch.Size) -> torch.Tensor: ...

    def log_prob(self, value: torch.Tensor) -> torch.Tensor: ...
