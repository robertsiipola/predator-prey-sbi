from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
import torch

from predator_prey_sbi.features import summarize_series
from predator_prey_sbi.types import PosteriorLike
from predator_prey_sbi.simulator import simulate_lv


Simulator = Callable[[torch.Tensor], torch.Tensor]


def build_prior(
    prior_config: dict[str, Any],
    parameter_order: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    lows: list[float] = []
    highs: list[float] = []
    for name in parameter_order:
        if name not in prior_config:
            raise ValueError(f"Missing prior range for parameter: {name}")
        low, high = prior_config[name]
        lows.append(float(low))
        highs.append(float(high))
    low_tensor = torch.tensor(lows, dtype=torch.float32)
    high_tensor = torch.tensor(highs, dtype=torch.float32)
    return low_tensor, high_tensor


def build_simulator(
    years: list[float],
    x0: tuple[float, float],
    dt: float,
    noise_scale: float,
    parameter_order: list[str],
) -> Simulator:
    def simulator(theta: torch.Tensor) -> torch.Tensor:
        theta_np = theta.detach().cpu().numpy().astype(float)
        if theta_np.shape[0] != len(parameter_order):
            raise ValueError("Theta dimension does not match parameter order")
        values = {name: float(theta_np[idx]) for idx, name in enumerate(parameter_order)}
        params = {
            "alpha": values["alpha"],
            "beta": values["beta"],
            "delta": values["delta"],
            "gamma": values["gamma"],
        }
        if "hare0" in values and "lynx0" in values:
            sim_x0 = (values["hare0"], values["lynx0"])
        else:
            sim_x0 = x0
        hare_sim, lynx_sim = simulate_lv(
            years=years,
            params=params,
            x0=sim_x0,
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
    sample_with: str,
    mcmc_method: str,
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
    posterior = cast(
        PosteriorLike,
        inference.build_posterior(
            density_estimator=density_estimator,
            sample_with=sample_with,
            mcmc_method=mcmc_method,
        ),
    )

    return posterior, theta, x
