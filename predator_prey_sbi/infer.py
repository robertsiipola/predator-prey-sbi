from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from predator_prey_sbi.config import load_config
from predator_prey_sbi.data import load_lynx_hare
from predator_prey_sbi.features import summarize_series
from predator_prey_sbi.runtime import configure_runtime
from predator_prey_sbi.simulator import simulate_lv


def _make_run_dir(base_dir: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_prior(prior_config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_low, alpha_high = prior_config["alpha"]
    beta_low, beta_high = prior_config["beta"]
    delta_low, delta_high = prior_config["delta"]
    gamma_low, gamma_high = prior_config["gamma"]
    low = torch.tensor([alpha_low, beta_low, delta_low, gamma_low], dtype=torch.float32)
    high = torch.tensor(
        [alpha_high, beta_high, delta_high, gamma_high], dtype=torch.float32
    )
    return low, high


def infer_from_file(config_path: str, observed_path: str) -> str:
    configure_runtime()
    config = load_config(config_path)

    data_path = observed_path or str(config["data_path"])
    years, hare_obs, lynx_obs = load_lynx_hare(data_path)

    inference_cfg = config.get("inference", {})
    num_simulations = int(inference_cfg.get("num_simulations", 500))
    num_samples = int(inference_cfg.get("num_samples", 500))
    num_workers = int(inference_cfg.get("num_workers", 1))
    seed = inference_cfg.get("seed", 0)
    noise_scale = float(inference_cfg.get("noise_scale", 0.0))
    prior_cfg = inference_cfg.get("prior", {})

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    use_observed_initial = bool(config.get("use_observed_initial", True))
    if use_observed_initial:
        x0 = (hare_obs[0], lynx_obs[0])
    else:
        x0_values = config.get("x0", [10.0, 10.0])
        x0 = (float(x0_values[0]), float(x0_values[1]))

    dt = float(config.get("dt", 0.1))

    summary_obs = summarize_series(hare_obs, lynx_obs)
    x_o = torch.tensor(summary_obs, dtype=torch.float32).unsqueeze(0)

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

    from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
    from sbi.utils import BoxUniform

    prior_low, prior_high = _build_prior(prior_cfg)
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
    posterior = inference.build_posterior(density_estimator)

    samples = posterior.sample((num_samples,), x=x_o)

    output_dir = str(inference_cfg.get("output_dir", config.get("output_dir", "runs")))
    run_dir = _make_run_dir(output_dir)
    output_path = run_dir / "posterior_samples.npz"
    np.savez(
        output_path,
        alpha=samples[:, 0].detach().cpu().numpy(),
        beta=samples[:, 1].detach().cpu().numpy(),
        delta=samples[:, 2].detach().cpu().numpy(),
        gamma=samples[:, 3].detach().cpu().numpy(),
    )

    print(
        "Training neural posterior estimator with sbi on {n} simulations".format(
            n=num_simulations
        )
    )
    print(f"Saved posterior samples to {output_path}")

    return str(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run neural SBI inference.")
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
    args = parser.parse_args()
    infer_from_file(args.config, args.observed)


if __name__ == "__main__":
    main()
