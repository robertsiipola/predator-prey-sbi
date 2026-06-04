from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import torch

from predator_prey_sbi.embedding import build_embedding_net
from predator_prey_sbi.features import build_embedding_input, summarize_series
from predator_prey_sbi.parameters import (
    resolve_initial_conditions,
    resolve_lv_params,
    resolve_observation_ar1,
    resolve_noise_scales,
    resolve_observation_lag,
    resolve_observation_power,
    resolve_observation_scales,
    resolve_process_noise_correlation,
    resolve_process_noise_scale,
)
from predator_prey_sbi.types import PosteriorLike
from predator_prey_sbi.simulator import simulate_lv


Simulator = Callable[[torch.Tensor], torch.Tensor]

if TYPE_CHECKING:
    from sbi.inference.posteriors.direct_posterior import DirectPosterior


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
    process_noise_scale: float,
    parameter_order: list[str],
    feature_mode: str = "summary",
    observation_operator: str = "point",
    observation_substeps: int = 10,
    embedding_transform: str = "log1p",
) -> Simulator:
    mode = feature_mode.lower()

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        theta_np = theta.detach().cpu().numpy().astype(float)
        if theta_np.shape[0] != len(parameter_order):
            raise ValueError("Theta dimension does not match parameter order")
        values = {
            name: float(theta_np[idx]) for idx, name in enumerate(parameter_order)
        }
        params = resolve_lv_params(values)
        sim_x0 = resolve_initial_conditions(values, x0)
        sim_noise = resolve_noise_scales(values, noise_scale)
        sim_process_noise = resolve_process_noise_scale(values, process_noise_scale)
        sim_process_noise_corr = resolve_process_noise_correlation(values, 0.0)
        sim_obs_scale = resolve_observation_scales(values)
        sim_obs_power = resolve_observation_power(values)
        sim_obs_lag = resolve_observation_lag(values, 0.0)
        sim_obs_ar1 = resolve_observation_ar1(values, (0.0, 0.0))
        obs_ref: tuple[float, float] | None = None
        if sim_obs_power != (1.0, 1.0) and {"log_x_eq", "log_y_eq"}.issubset(values):
            obs_ref = (
                float(np.exp(float(values["log_x_eq"]))),
                float(np.exp(float(values["log_y_eq"]))),
            )
        hare_sim, lynx_sim = simulate_lv(
            years=years,
            params=params,
            x0=sim_x0,
            dt=dt,
            noise_scale=sim_noise,
            process_noise_scale=sim_process_noise,
            process_noise_correlation=sim_process_noise_corr,
            rng_seed=None,
            observation_scale=sim_obs_scale,
            observation_power=sim_obs_power,
            observation_reference=obs_ref,
            observation_operator=observation_operator,
            observation_substeps=observation_substeps,
            observation_lag=sim_obs_lag,
            observation_ar1=sim_obs_ar1,
        )
        if mode == "embedding":
            features = build_embedding_input(
                hare_sim, lynx_sim, transform=embedding_transform
            )
        else:
            features = np.asarray(
                summarize_series(hare_sim, lynx_sim), dtype=np.float32
            )
        return torch.tensor(features, dtype=torch.float32)

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
    feature_mode: str = "summary",
    embedding_config: dict[str, Any] | None = None,
) -> tuple[PosteriorLike, torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
    from sbi.utils import BoxUniform
    from sbi.utils import get_nn_models

    prior = BoxUniform(low=prior_low, high=prior_high)
    prepared_simulator, prepared_prior = prepare_for_sbi(simulator, prior)
    mode = feature_mode.lower()
    if mode == "embedding":
        embed_cfg = embedding_config or {}
        embedding_net = build_embedding_net(embed_cfg)
        density_estimator = get_nn_models.posterior_nn(
            model=str(embed_cfg.get("model", "nsf")),
            z_score_x=str(embed_cfg.get("z_score_x", "structured")),
            hidden_features=int(embed_cfg.get("hidden_features", 50)),
            num_transforms=int(embed_cfg.get("num_transforms", 5)),
            num_bins=int(embed_cfg.get("num_bins", 10)),
            embedding_net=embedding_net,
        )
        inference = SNPE(prior=prepared_prior, density_estimator=density_estimator)
    else:
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


def train_posterior_sequential(
    simulator: Simulator,
    prior_low: torch.Tensor,
    prior_high: torch.Tensor,
    x_o: torch.Tensor,
    num_simulations: int,
    num_rounds: int,
    num_workers: int,
    seed: int | None,
    sample_with: str,
    mcmc_method: str,
    prior_mixin: float,
    simulations_per_round: list[int] | None = None,
    feature_mode: str = "summary",
    embedding_config: dict[str, Any] | None = None,
) -> tuple[PosteriorLike, torch.Tensor, torch.Tensor]:
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")
    if not (0.0 <= prior_mixin < 1.0):
        raise ValueError("prior_mixin must be in [0, 1)")

    if simulations_per_round is None:
        per_round = [num_simulations // num_rounds] * num_rounds
        per_round[-1] += num_simulations - sum(per_round)
    else:
        per_round = [int(x) for x in simulations_per_round]
        if len(per_round) != num_rounds:
            raise ValueError("simulations_per_round length must match num_rounds")
        if any(n <= 0 for n in per_round):
            raise ValueError("simulations_per_round entries must be positive")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
    from sbi.utils import BoxUniform
    from sbi.utils import get_nn_models

    prior = BoxUniform(low=prior_low, high=prior_high)
    prepared_simulator, prepared_prior = prepare_for_sbi(simulator, prior)
    mode = feature_mode.lower()
    if mode == "embedding":
        embed_cfg = embedding_config or {}
        embedding_net = build_embedding_net(embed_cfg)
        density_estimator = get_nn_models.posterior_nn(
            model=str(embed_cfg.get("model", "nsf")),
            z_score_x=str(embed_cfg.get("z_score_x", "structured")),
            hidden_features=int(embed_cfg.get("hidden_features", 50)),
            num_transforms=int(embed_cfg.get("num_transforms", 5)),
            num_bins=int(embed_cfg.get("num_bins", 10)),
            embedding_net=embedding_net,
        )
        inference = SNPE(prior=prepared_prior, density_estimator=density_estimator)
    else:
        inference = SNPE(prior=prepared_prior)

    theta = torch.empty((0,))
    x = torch.empty((0,))
    posterior: PosteriorLike | None = None
    density_estimator: Any | None = None
    for round_idx in range(num_rounds):
        if round_idx == 0:
            theta, x = simulate_for_sbi(
                prepared_simulator,
                proposal=prepared_prior,
                num_simulations=per_round[round_idx],
                num_workers=num_workers,
            )
            density_estimator = inference.append_simulations(theta, x).train()
        else:
            if density_estimator is None:
                raise RuntimeError("Sequential SNPE missing density estimator state")

            proposal_posterior = cast(
                PosteriorLike,
                inference.build_posterior(
                    density_estimator=density_estimator,
                    sample_with="rejection",
                    mcmc_method="slice_np",
                ),
            )
            posterior_for_proposal = cast(
                "DirectPosterior", proposal_posterior.set_default_x(x_o)
            )

            num_this_round = per_round[round_idx]
            num_prior = int(round(num_this_round * prior_mixin))
            num_post = num_this_round - num_prior

            theta_batches: list[torch.Tensor] = []
            x_batches: list[torch.Tensor] = []

            if num_post > 0:
                theta_post, x_post = simulate_for_sbi(
                    prepared_simulator,
                    proposal=posterior_for_proposal,
                    num_simulations=num_post,
                    num_workers=num_workers,
                )
                inference.append_simulations(
                    theta_post, x_post, proposal=posterior_for_proposal
                )
                theta_batches.append(theta_post)
                x_batches.append(x_post)

            if num_prior > 0:
                theta_prior, x_prior = simulate_for_sbi(
                    prepared_simulator,
                    proposal=prepared_prior,
                    num_simulations=num_prior,
                    num_workers=num_workers,
                )
                inference.append_simulations(theta_prior, x_prior)
                theta_batches.append(theta_prior)
                x_batches.append(x_prior)

            theta = torch.cat(theta_batches, dim=0)
            x = torch.cat(x_batches, dim=0)
            density_estimator = inference.train()

        if round_idx == num_rounds - 1:
            if density_estimator is None:
                raise RuntimeError("Sequential SNPE missing final density estimator")
            posterior = cast(
                PosteriorLike,
                inference.build_posterior(
                    density_estimator=density_estimator,
                    sample_with=sample_with,
                    mcmc_method=mcmc_method,
                ),
            )

    if posterior is None:
        raise RuntimeError("Sequential SNPE did not produce a posterior")
    return posterior, theta, x
