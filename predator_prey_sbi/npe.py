from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
import torch

from predator_prey_sbi.features import summarize_series
from predator_prey_sbi.types import PosteriorLike
from predator_prey_sbi.simulator import simulate_lv


Simulator = Callable[[torch.Tensor], torch.Tensor]


def build_prior(prior_config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_low, alpha_high = prior_config["alpha"]
    beta_low, beta_high = prior_config["beta"]
    delta_low, delta_high = prior_config["delta"]
    gamma_low, gamma_high = prior_config["gamma"]
    low = torch.tensor([alpha_low, beta_low, delta_low, gamma_low], dtype=torch.float32)
    high = torch.tensor(
        [alpha_high, beta_high, delta_high, gamma_high], dtype=torch.float32
    )
    return low, high


def build_simulator(
    years: list[float],
    x0: tuple[float, float],
    dt: float,
    noise_scale: float,
) -> Simulator:
    def simulator(theta: torch.Tensor) -> torch.Tensor:
        theta_np = theta.detach().cpu().numpy().astype(float)
        params = {
            "alpha": float(theta_np[0]),
            "beta": float(theta_np[1]),
            "delta": float(theta_np[2]),
            "gamma": float(theta_np[3]),
        }
        hare_sim, lynx_sim = simulate_lv(
            years=years,
            params=params,
            x0=x0,
            dt=dt,
            noise_scale=noise_scale,
            rng_seed=None,
        )
        summary = summarize_series(hare_sim, lynx_sim)
        return torch.tensor(summary, dtype=torch.float32)

    return simulator


def train_posterior(
    simulator: Simulator,
    prior_low: torch.Tensor,
    prior_high: torch.Tensor,
    num_simulations: int,
    num_workers: int,
    seed: int | None,
) -> tuple[PosteriorLike, torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=prior_low, high=prior_high)
    prepared_simulator, prepared_prior = prepare_for_sbi(simulator, prior)

    inference = SNPE(prior=prepared_prior)
    theta, x = simulate_for_sbi(
        prepared_simulator,
        proposal=prepared_prior,
        num_simulations=num_simulations,
        num_workers=num_workers,
    )
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = cast(PosteriorLike, inference.build_posterior(density_estimator))

    return posterior, theta, x
