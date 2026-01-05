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
    required = {"alpha", "beta", "delta", "gamma"}
    if not required.issubset(samples):
        missing = required.difference(samples)
        raise ValueError(f"Posterior samples missing keys: {sorted(missing)}")
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
        params_list.append({key: float(values[idx]) for key, values in posterior_samples.items()})
    return params_list


def posterior_predictive(
    posterior_samples: dict[str, np.ndarray],
    years: list[float],
    obs: tuple[list[float], list[float]],
    x0: tuple[float, float],
    dt: float,
    noise_scale: float,
    n_draws: int,
    seed: int | None,
    output_dir: Path,
) -> dict[str, float | str]:
    params_list = _sample_posterior_params(posterior_samples, n_draws, seed)

    hare_obs, lynx_obs = obs
    hare_sims = []
    lynx_sims = []

    from predator_prey_sbi.simulator import simulate_lv

    for params in params_list:
        sim_x0 = x0
        if "hare0" in params and "lynx0" in params:
            sim_x0 = (params["hare0"], params["lynx0"])
        hare_sim, lynx_sim = simulate_lv(
            years=years,
            params={
                "alpha": params["alpha"],
                "beta": params["beta"],
                "delta": params["delta"],
                "gamma": params["gamma"],
            },
            x0=sim_x0,
            dt=dt,
            noise_scale=noise_scale,
            rng_seed=None,
        )
        hare_sims.append(hare_sim)
        lynx_sims.append(lynx_sim)

    hare_arr = np.asarray(hare_sims)
    lynx_arr = np.asarray(lynx_sims)

    hare_q05, hare_q50, hare_q95 = np.percentile(hare_arr, [5, 50, 95], axis=0)
    lynx_q05, lynx_q50, lynx_q95 = np.percentile(lynx_arr, [5, 50, 95], axis=0)

    hare_rmse = float(np.sqrt(np.mean((hare_q50 - np.asarray(hare_obs)) ** 2)))
    lynx_rmse = float(np.sqrt(np.mean((lynx_q50 - np.asarray(lynx_obs)) ** 2)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].fill_between(years, hare_q05, hare_q95, color="tab:blue", alpha=0.2)
    axes[0].plot(years, hare_q50, color="tab:blue", label="Posterior median")
    axes[0].plot(years, hare_obs, color="tab:blue", linestyle="--", label="Observed")
    axes[0].set_ylabel("Hare")
    axes[0].legend(loc="upper right")

    axes[1].fill_between(years, lynx_q05, lynx_q95, color="tab:orange", alpha=0.2)
    axes[1].plot(years, lynx_q50, color="tab:orange", label="Posterior median")
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
        "posterior_predictive_hare_rmse": hare_rmse,
        "posterior_predictive_lynx_rmse": lynx_rmse,
        "posterior_predictive_plot": str(plot_path),
    }


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
        n_draws = int(diagnostics_cfg.get("posterior_draws", 200))
        noise_scale = float(diagnostics_cfg.get("noise_scale", 0.0))
        seed = diagnostics_cfg.get("seed", 0)

        metrics.update(
            posterior_predictive(
                posterior_samples=posterior_samples,
                years=years,
                obs=(hare_obs, lynx_obs),
                x0=x0,
                dt=dt,
                noise_scale=noise_scale,
                n_draws=n_draws,
                seed=seed,
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
    parameter_order = list(
        inference_cfg.get("parameter_order", ["alpha", "beta", "delta", "gamma"])
    )

    use_observed_initial = bool(config.get("use_observed_initial", True))
    if use_observed_initial:
        x0 = (hare_obs[0], lynx_obs[0])
    else:
        x0_values = config.get("x0", [10.0, 10.0])
        x0 = (float(x0_values[0]), float(x0_values[1]))

    dt = float(config.get("dt", 0.1))
    simulator = build_simulator(
        years=years,
        x0=x0,
        dt=dt,
        noise_scale=float(inference_cfg.get("noise_scale", 0.0)),
        parameter_order=parameter_order,
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
