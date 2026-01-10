from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from predator_prey_sbi.config import load_config
from predator_prey_sbi.data import load_lynx_hare
from predator_prey_sbi.npe import build_prior, build_simulator, train_posterior
from predator_prey_sbi.parameters import (
    resolve_initial_conditions,
    resolve_lv_params,
    resolve_noise_scales,
    resolve_observation_lag,
    resolve_observation_scales,
    resolve_process_noise_scale,
)
from predator_prey_sbi.priors import build_structure_aware_prior
from predator_prey_sbi.runtime import configure_runtime
from predator_prey_sbi.types import PosteriorLike, PriorLike


def _make_run_dir(base_dir: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_posterior_samples(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    samples = {key: data[key] for key in data.files}
    base_required = {"alpha", "beta", "delta", "gamma"}
    reparam_required = {"alpha", "gamma", "x_star", "y_star"}
    structure_base = {"log_T", "log_r", "log_x_eq", "log_y_eq"}
    structure_required_a = structure_base | {"log_k_ratio"}
    structure_required_b = structure_base | {"log_tau_damp"}
    if (
        base_required.issubset(samples)
        or reparam_required.issubset(samples)
        or structure_required_a.issubset(samples)
        or structure_required_b.issubset(samples)
    ):
        return samples
    missing_base = sorted(base_required.difference(samples))
    missing_reparam = sorted(reparam_required.difference(samples))
    missing_structure_a = sorted(structure_required_a.difference(samples))
    missing_structure_b = sorted(structure_required_b.difference(samples))
    raise ValueError(
        "Posterior samples missing keys for either parameterization. "
        "Base missing: {base}; reparam missing: {reparam}; "
        "structure(k_ratio) missing: {structure_a}; structure(tau_damp) missing: {structure_b}".format(
            base=missing_base,
            reparam=missing_reparam,
            structure_a=missing_structure_a,
            structure_b=missing_structure_b,
        )
    )
    return samples


def _sample_posterior_params(
    posterior_samples: dict[str, np.ndarray],
    num_draws: int,
    seed: int | None,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    sample_count = len(next(iter(posterior_samples.values())))
    if sample_count == 0:
        raise ValueError("Posterior samples are empty")
    replace = num_draws > sample_count
    indices = rng.choice(sample_count, size=num_draws, replace=replace)
    params_list: list[dict[str, float]] = []
    for idx in indices:
        params_list.append(
            {key: float(values[idx]) for key, values in posterior_samples.items()}
        )
    return params_list


def posterior_predictive(
    posterior_samples: dict[str, np.ndarray],
    years: list[float],
    obs: tuple[list[float], list[float]],
    x0: tuple[float, float],
    dt: float,
    noise_scale: float,
    process_noise_scale: float,
    n_draws: int,
    seed: int | None,
    observation_operator: str,
    observation_substeps: int,
    output_dir: Path,
) -> dict[str, float | str]:
    params_list = _sample_posterior_params(posterior_samples, n_draws, seed)
    noise_seed_rng = np.random.default_rng(seed)

    hare_obs, lynx_obs = obs
    hare_sims = []
    lynx_sims = []

    from predator_prey_sbi.simulator import simulate_lv

    for params in params_list:
        sim_x0 = resolve_initial_conditions(params, x0)
        sim_noise = resolve_noise_scales(params, noise_scale)
        sim_process_noise = resolve_process_noise_scale(params, process_noise_scale)
        sim_obs_scale = resolve_observation_scales(params)
        sim_obs_lag = resolve_observation_lag(params, 0.0)
        hare_sim, lynx_sim = simulate_lv(
            years=years,
            params=resolve_lv_params(params),
            x0=sim_x0,
            dt=dt,
            noise_scale=sim_noise,
            process_noise_scale=sim_process_noise,
            rng_seed=int(noise_seed_rng.integers(0, 2**32 - 1)),
            observation_scale=sim_obs_scale,
            observation_operator=observation_operator,
            observation_substeps=observation_substeps,
            observation_lag=sim_obs_lag,
        )
        hare_sims.append(hare_sim)
        lynx_sims.append(lynx_sim)

    hare_arr = np.asarray(hare_sims)
    lynx_arr = np.asarray(lynx_sims)

    hare_q05, hare_q50, hare_q95 = np.percentile(hare_arr, [5, 50, 95], axis=0)
    lynx_q05, lynx_q50, lynx_q95 = np.percentile(lynx_arr, [5, 50, 95], axis=0)

    hare_mean = np.mean(hare_arr, axis=0)
    lynx_mean = np.mean(lynx_arr, axis=0)

    hare_rmse_mean = float(np.sqrt(np.mean((hare_mean - np.asarray(hare_obs)) ** 2)))
    lynx_rmse_mean = float(np.sqrt(np.mean((lynx_mean - np.asarray(lynx_obs)) ** 2)))
    hare_rmse_median = float(np.sqrt(np.mean((hare_q50 - np.asarray(hare_obs)) ** 2)))
    lynx_rmse_median = float(np.sqrt(np.mean((lynx_q50 - np.asarray(lynx_obs)) ** 2)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].fill_between(years, hare_q05, hare_q95, color="tab:blue", alpha=0.2)
    axes[0].plot(years, hare_q50, color="tab:blue", label="Posterior median")
    axes[0].plot(
        years, hare_mean, color="tab:blue", linestyle=":", label="Posterior mean"
    )
    axes[0].plot(years, hare_obs, color="tab:blue", linestyle="--", label="Observed")
    axes[0].set_ylabel("Hare")
    axes[0].legend(loc="upper right")

    axes[1].fill_between(years, lynx_q05, lynx_q95, color="tab:orange", alpha=0.2)
    axes[1].plot(years, lynx_q50, color="tab:orange", label="Posterior median")
    axes[1].plot(
        years, lynx_mean, color="tab:orange", linestyle=":", label="Posterior mean"
    )
    axes[1].plot(years, lynx_obs, color="tab:orange", linestyle="--", label="Observed")
    axes[1].set_ylabel("Lynx")
    axes[1].set_xlabel("Year")
    axes[1].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "posterior_predictive.png"
    fig.savefig(plot_path)
    plt.close(fig)

    return {
        "posterior_predictive_hare_rmse": hare_rmse_mean,
        "posterior_predictive_lynx_rmse": lynx_rmse_mean,
        "posterior_predictive_hare_rmse_median": hare_rmse_median,
        "posterior_predictive_lynx_rmse_median": lynx_rmse_median,
        "posterior_predictive_plot": str(plot_path),
    }


def _tau_damp_years_from_params(params: dict[str, float]) -> float | None:
    if {"log_tau_damp"}.issubset(params):
        tau = float(np.exp(float(params["log_tau_damp"])))
        if not np.isfinite(tau) or tau <= 0:
            return None
        return tau
    required = {"log_k_ratio", "log_T", "log_r"}
    if not required.issubset(params):
        return None
    k_ratio = float(np.exp(float(params["log_k_ratio"])))
    t = float(np.exp(float(params["log_T"])))
    r = float(np.exp(float(params["log_r"])))
    if not (np.isfinite(k_ratio) and np.isfinite(t) and np.isfinite(r)):
        return None
    if k_ratio <= 0 or t <= 0 or r <= 0:
        return None
    return (k_ratio * t) / (np.pi * r)


def latent_dynamics_diagnostic(
    posterior_samples: dict[str, np.ndarray],
    years: list[float],
    x0: tuple[float, float],
    dt: float,
    process_noise_scale: float,
    n_draws: int,
    n_plot: int,
    seed: int | None,
    observation_operator: str,
    observation_substeps: int,
    output_dir: Path,
) -> dict[str, float | str]:
    params_list = _sample_posterior_params(posterior_samples, n_draws, seed)
    rng = np.random.default_rng(seed)

    from predator_prey_sbi.simulator import simulate_lv

    hare_latents: list[list[float]] = []
    lynx_latents: list[list[float]] = []
    tau_damps: list[float] = []
    for params in params_list:
        sim_x0 = resolve_initial_conditions(params, x0)
        sim_process_noise = resolve_process_noise_scale(params, process_noise_scale)
        sim_obs_lag = resolve_observation_lag(params, 0.0)
        hare_latent, lynx_latent = simulate_lv(
            years=years,
            params=resolve_lv_params(params),
            x0=sim_x0,
            dt=dt,
            noise_scale=0.0,
            process_noise_scale=sim_process_noise,
            rng_seed=int(rng.integers(0, 2**32 - 1)),
            observation_scale=(1.0, 1.0),
            observation_operator=observation_operator,
            observation_substeps=observation_substeps,
            observation_lag=sim_obs_lag,
        )
        hare_latents.append(hare_latent)
        lynx_latents.append(lynx_latent)
        tau = _tau_damp_years_from_params(params)
        if tau is not None:
            tau_damps.append(float(tau))

    hare_arr = np.asarray(hare_latents, dtype=float)
    lynx_arr = np.asarray(lynx_latents, dtype=float)

    hare_mean = np.mean(hare_arr, axis=0)
    lynx_mean = np.mean(lynx_arr, axis=0)

    per_draw_hare_std = np.std(hare_arr, axis=1)
    per_draw_lynx_std = np.std(lynx_arr, axis=1)
    mean_hare_draw_std = float(np.mean(per_draw_hare_std))
    mean_lynx_draw_std = float(np.mean(per_draw_lynx_std))
    hare_mean_std = float(np.std(hare_mean))
    lynx_mean_std = float(np.std(lynx_mean))
    hare_phase_coherence = float(hare_mean_std / (mean_hare_draw_std + 1e-12))
    lynx_phase_coherence = float(lynx_mean_std / (mean_lynx_draw_std + 1e-12))

    n_t = hare_arr.shape[1]
    mid = max(1, n_t // 2)
    hare_damp_ratio = float(
        np.median(
            np.std(hare_arr[:, mid:], axis=1)
            / (np.std(hare_arr[:, :mid], axis=1) + 1e-12)
        )
    )
    lynx_damp_ratio = float(
        np.median(
            np.std(lynx_arr[:, mid:], axis=1)
            / (np.std(lynx_arr[:, :mid], axis=1) + 1e-12)
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    n_plot_clipped = max(1, min(n_plot, hare_arr.shape[0]))
    for idx in range(n_plot_clipped):
        axes[0].plot(years, hare_arr[idx], color="tab:blue", alpha=0.25, linewidth=1.0)
        axes[1].plot(
            years, lynx_arr[idx], color="tab:orange", alpha=0.25, linewidth=1.0
        )
    axes[0].plot(
        years, hare_mean, color="tab:blue", linewidth=2.0, label="Mean over draws"
    )
    axes[1].plot(
        years, lynx_mean, color="tab:orange", linewidth=2.0, label="Mean over draws"
    )
    axes[0].set_ylabel("Hare (latent annual)")
    axes[1].set_ylabel("Lynx (latent annual)")
    axes[1].set_xlabel("Year")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "latent_posterior_draws.png"
    fig.savefig(plot_path)
    plt.close(fig)

    out: dict[str, float | str] = {
        "latent_draws_plot": str(plot_path),
        "latent_phase_coherence_hare": hare_phase_coherence,
        "latent_phase_coherence_lynx": lynx_phase_coherence,
        "latent_damping_ratio_hare": hare_damp_ratio,
        "latent_damping_ratio_lynx": lynx_damp_ratio,
    }
    if tau_damps:
        tau_arr = np.asarray(tau_damps, dtype=float)
        record_years = float(years[-1] - years[0])
        out.update(
            {
                "tau_damp_years_p10": float(np.percentile(tau_arr, 10.0)),
                "tau_damp_years_p50": float(np.percentile(tau_arr, 50.0)),
                "tau_damp_years_p90": float(np.percentile(tau_arr, 90.0)),
                "tau_damp_frac_lt_record": float(np.mean(tau_arr < record_years)),
            }
        )
    return out


def _sbc_rank_histogram(
    ranks: dict[str, list[int]],
    num_samples: int,
    parameter_order: list[str],
    output_path: Path,
) -> None:
    n_params = len(parameter_order)
    ncols = 2
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes_list = np.ravel(axes)
    for ax, param in zip(axes_list, parameter_order, strict=False):
        ax.hist(ranks[param], bins=10, range=(0, num_samples), color="tab:gray")
        ax.set_title(param)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Count")
    for ax in axes_list[n_params:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_sbc(
    posterior: PosteriorLike,
    prior: PriorLike,
    simulator: Any,
    num_datasets: int,
    num_posterior_samples: int,
    seed: int | None,
    output_dir: Path,
    parameter_order: list[str],
) -> dict[str, Any]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ranks = {param: [] for param in parameter_order}
    coverage_counts = {param: 0 for param in parameter_order}

    for _ in range(num_datasets):
        theta_true = prior.sample((1,))
        x = simulator(theta_true[0]).unsqueeze(0)
        posterior_samples = posterior.sample((num_posterior_samples,), x=x)

        for idx, param in enumerate(parameter_order):
            samples = posterior_samples[:, idx].detach().cpu().numpy()
            true_val = float(theta_true[0, idx].detach().cpu().numpy())
            rank = int(np.sum(samples < true_val))
            ranks[param].append(rank)

            lower = np.percentile(samples, 5)
            upper = np.percentile(samples, 95)
            if lower <= true_val <= upper:
                coverage_counts[param] += 1

    coverage = {
        param: coverage_counts[param] / max(1, num_datasets)
        for param in coverage_counts
    }

    hist_path = output_dir / "sbc_rank_histogram.png"
    _sbc_rank_histogram(ranks, num_posterior_samples, parameter_order, hist_path)

    return {
        "sbc_rank_histogram": str(hist_path),
        "sbc_coverage": coverage,
        "sbc_num_datasets": num_datasets,
        "sbc_num_posterior_samples": num_posterior_samples,
    }


def diagnostics_from_file(
    config_path: str,
    observed_path: str,
    posterior_path: str,
) -> str:
    configure_runtime()
    config = load_config(config_path)

    data_path = observed_path or str(config["data_path"])
    years, hare_obs, lynx_obs = load_lynx_hare(data_path)

    diagnostics_cfg = config.get("diagnostics", {})
    output_dir = str(
        diagnostics_cfg.get("output_dir", config.get("output_dir", "runs"))
    )
    run_dir = _make_run_dir(output_dir)

    metrics: dict[str, Any] = {}

    if posterior_path:
        posterior_samples = _load_posterior_samples(posterior_path)
        use_observed_initial = bool(config.get("use_observed_initial", True))
        if use_observed_initial:
            x0 = (hare_obs[0], lynx_obs[0])
        else:
            x0_values = config.get("x0", [10.0, 10.0])
            x0 = (float(x0_values[0]), float(x0_values[1]))

        dt = float(config.get("dt", 0.1))
        observation_operator = str(config.get("observation_operator", "point"))
        observation_substeps = int(config.get("observation_substeps", 10))
        n_draws = int(diagnostics_cfg.get("posterior_draws", 200))
        noise_scale = float(diagnostics_cfg.get("noise_scale", 0.0))
        process_noise_scale = float(
            diagnostics_cfg.get(
                "process_noise_scale",
                config.get("inference", {}).get("process_noise_scale", 0.0),
            )
        )
        seed = diagnostics_cfg.get("seed", 0)

        metrics.update(
            posterior_predictive(
                posterior_samples=posterior_samples,
                years=years,
                obs=(hare_obs, lynx_obs),
                x0=x0,
                dt=dt,
                noise_scale=noise_scale,
                process_noise_scale=process_noise_scale,
                n_draws=n_draws,
                seed=seed,
                observation_operator=observation_operator,
                observation_substeps=observation_substeps,
                output_dir=run_dir,
            )
        )

        latent_cfg = diagnostics_cfg.get("latent_diagnostics", {})
        latent_enabled = bool(latent_cfg.get("enabled", False))
        if latent_enabled:
            latent_draws = int(latent_cfg.get("num_draws", n_draws))
            latent_plot = int(latent_cfg.get("num_plot", 30))
            metrics.update(
                latent_dynamics_diagnostic(
                    posterior_samples=posterior_samples,
                    years=years,
                    x0=x0,
                    dt=dt,
                    process_noise_scale=process_noise_scale,
                    n_draws=latent_draws,
                    n_plot=latent_plot,
                    seed=seed,
                    observation_operator=observation_operator,
                    observation_substeps=observation_substeps,
                    output_dir=run_dir,
                )
            )
    else:
        metrics["posterior_predictive_skipped"] = True

    sbc_cfg = diagnostics_cfg.get("sbc", {})
    sbc_num_simulations = int(
        sbc_cfg.get(
            "num_simulations", config.get("inference", {}).get("num_simulations", 500)
        )
    )
    sbc_num_datasets = int(sbc_cfg.get("num_datasets", 20))
    sbc_num_samples = int(sbc_cfg.get("num_posterior_samples", 200))
    sbc_seed = sbc_cfg.get("seed", diagnostics_cfg.get("seed", 0))
    sbc_num_workers = int(
        sbc_cfg.get("num_workers", config.get("inference", {}).get("num_workers", 1))
    )
    sbc_sample_with = str(sbc_cfg.get("sample_with", "rejection"))
    sbc_mcmc_method = str(sbc_cfg.get("mcmc_method", "slice_np"))
    inference_cfg = config.get("inference", {})
    prior_cfg = inference_cfg.get("prior", {})
    feature_mode = str(inference_cfg.get("feature_mode", "summary"))
    embedding_cfg = (
        inference_cfg.get("embedding", {}) if isinstance(inference_cfg, dict) else {}
    )
    embedding_transform = str(inference_cfg.get("embedding_transform", "log1p"))
    prior_scheme = str(inference_cfg.get("prior_scheme", "default"))
    k_parameterization = str(inference_cfg.get("k_parameterization", "k_ratio"))
    include_process_noise = bool(inference_cfg.get("include_process_noise", False))
    include_holling = bool(inference_cfg.get("include_holling", False))
    include_observation_scale = bool(
        inference_cfg.get("include_observation_scale", False)
    )
    include_observation_lag = bool(inference_cfg.get("include_observation_lag", False))
    parameter_order = list(
        inference_cfg.get(
            "parameter_order",
            [
                "log_T",
                "log_r",
                "log_x_eq",
                "log_y_eq",
                "log_k_ratio",
                "eps_h0",
                "eps_l0",
                "log_sigma_h",
                "log_sigma_l",
            ],
        )
    )

    use_observed_initial = bool(config.get("use_observed_initial", True))
    if use_observed_initial:
        x0 = (hare_obs[0], lynx_obs[0])
    else:
        x0_values = config.get("x0", [10.0, 10.0])
        x0 = (float(x0_values[0]), float(x0_values[1]))

    dt = float(config.get("dt", 0.1))
    observation_operator = str(
        inference_cfg.get(
            "observation_operator", config.get("observation_operator", "point")
        )
    )
    observation_substeps = int(
        inference_cfg.get(
            "observation_substeps", config.get("observation_substeps", 10)
        )
    )
    process_noise_scale = float(inference_cfg.get("process_noise_scale", 0.0))

    if prior_scheme == "structure_aware":
        parameter_order, prior_cfg = build_structure_aware_prior(
            hare_obs,
            lynx_obs,
            prior_cfg,
            k_parameterization=k_parameterization,
            include_process_noise=include_process_noise,
            include_holling=include_holling,
            include_observation_scale=include_observation_scale,
            include_observation_lag=include_observation_lag,
        )
    simulator = build_simulator(
        years=years,
        x0=x0,
        dt=dt,
        noise_scale=float(inference_cfg.get("noise_scale", 0.0)),
        process_noise_scale=process_noise_scale,
        parameter_order=parameter_order,
        feature_mode=feature_mode,
        observation_operator=observation_operator,
        observation_substeps=observation_substeps,
        embedding_transform=embedding_transform,
    )
    prior_low, prior_high = build_prior(prior_cfg, parameter_order)

    posterior, _, _ = train_posterior(
        simulator=simulator,
        prior_low=prior_low,
        prior_high=prior_high,
        num_simulations=sbc_num_simulations,
        num_workers=sbc_num_workers,
        seed=sbc_seed,
        sample_with=sbc_sample_with,
        mcmc_method=sbc_mcmc_method,
        feature_mode=feature_mode,
        embedding_config=embedding_cfg,
    )

    from sbi.utils import BoxUniform

    prior = cast(PriorLike, BoxUniform(low=prior_low, high=prior_high))
    metrics.update(
        run_sbc(
            posterior=posterior,
            prior=prior,
            simulator=simulator,
            num_datasets=sbc_num_datasets,
            num_posterior_samples=sbc_num_samples,
            seed=sbc_seed,
            output_dir=run_dir,
            parameter_order=parameter_order,
        )
    )

    metrics_path = run_dir / "diagnostics_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Diagnostics metrics written to {metrics_path}")
    if "posterior_predictive_hare_rmse" in metrics:
        print(
            "Posterior predictive RMSE (hare): {hare:.3f}; (lynx): {lynx:.3f}".format(
                hare=metrics["posterior_predictive_hare_rmse"],
                lynx=metrics["posterior_predictive_lynx_rmse"],
            )
        )
    if "sbc_coverage" in metrics:
        print(f"SBC coverage: {metrics['sbc_coverage']}")
    return str(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBI diagnostics.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--observed",
        default="",
        help="Path to observed data; defaults to config data_path.",
    )
    parser.add_argument(
        "--posterior",
        default="",
        help="Path to posterior_samples.npz for posterior predictive checks.",
    )
    args = parser.parse_args()
    diagnostics_from_file(args.config, args.observed, args.posterior)


if __name__ == "__main__":
    main()
