from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

from predator_prey_sbi.config import load_config
from predator_prey_sbi.data import load_lynx_hare
from predator_prey_sbi.features import build_embedding_input, summarize_series
from predator_prey_sbi.npe import build_prior, build_simulator, train_posterior
from predator_prey_sbi.priors import build_structure_aware_prior
from predator_prey_sbi.runtime import configure_runtime


def _make_run_dir(base_dir: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    sample_with = str(inference_cfg.get("sample_with", "rejection"))
    mcmc_method = str(inference_cfg.get("mcmc_method", "slice_np"))
    feature_mode = str(inference_cfg.get("feature_mode", "summary"))
    embedding_cfg = (
        inference_cfg.get("embedding", {}) if isinstance(inference_cfg, dict) else {}
    )
    prior_scheme = str(inference_cfg.get("prior_scheme", "default"))
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

    if prior_scheme == "structure_aware":
        parameter_order, prior_cfg = build_structure_aware_prior(
            hare_obs, lynx_obs, prior_cfg
        )

    if feature_mode.lower() == "embedding":
        embedding_obs = build_embedding_input(hare_obs, lynx_obs)
        x_o = torch.tensor(embedding_obs, dtype=torch.float32).unsqueeze(0)
    else:
        summary_obs = summarize_series(hare_obs, lynx_obs)
        x_o = torch.tensor(summary_obs, dtype=torch.float32).unsqueeze(0)

    simulator = build_simulator(
        years=years,
        x0=x0,
        dt=dt,
        noise_scale=noise_scale,
        parameter_order=parameter_order,
        feature_mode=feature_mode,
    )

    prior_low, prior_high = build_prior(prior_cfg, parameter_order)
    posterior, _, _ = train_posterior(
        simulator=simulator,
        prior_low=prior_low,
        prior_high=prior_high,
        num_simulations=num_simulations,
        num_workers=num_workers,
        seed=seed,
        sample_with=sample_with,
        mcmc_method=mcmc_method,
        feature_mode=feature_mode,
        embedding_config=embedding_cfg,
    )

    samples = posterior.sample((num_samples,), x=x_o)

    output_dir = str(inference_cfg.get("output_dir", config.get("output_dir", "runs")))
    run_dir = _make_run_dir(output_dir)
    output_path = run_dir / "posterior_samples.npz"
    sample_arrays = {
        name: samples[:, idx].detach().cpu().numpy()
        for idx, name in enumerate(parameter_order)
    }
    np.savez(output_path, **sample_arrays)

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
