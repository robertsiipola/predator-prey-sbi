from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from predator_prey_sbi.data import load_lynx_hare
from predator_prey_sbi.features import summarize_series
from predator_prey_sbi.runtime import configure_runtime
from predator_prey_sbi.simulator import simulate_lv


def _make_run_dir(base_dir: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_smoke_test(
    rng_seed: int | None,
    num_simulations: int,
    num_samples: int,
    output_dir: str,
) -> Path:
    configure_runtime()

    if rng_seed is not None:
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)

    years, hare_obs, lynx_obs = load_lynx_hare("data/LynxHare.txt")
    summary_obs = summarize_series(hare_obs, lynx_obs)
    x_o = torch.tensor(summary_obs, dtype=torch.float32).unsqueeze(0)

    prior_low = torch.tensor([0.2, 0.05, 0.05, 0.2], dtype=torch.float32)
    prior_high = torch.tensor([3.0, 1.5, 1.5, 3.0], dtype=torch.float32)

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
            x0=(hare_obs[0], lynx_obs[0]),
            dt=0.1,
            noise_scale=0.0,
            rng_seed=None,
        )
        summary = summarize_series(hare_sim, lynx_sim)
        return torch.tensor(summary, dtype=torch.float32)

    from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=prior_low, high=prior_high)
    prepared_simulator, prepared_prior = prepare_for_sbi(simulator, prior)

    inference = SNPE(prior=prepared_prior)
    theta, x = simulate_for_sbi(
        prepared_simulator,
        proposal=prepared_prior,
        num_simulations=num_simulations,
        num_workers=1,
    )
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)

    samples = posterior.sample((num_samples,), x=x_o)
    run_dir = _make_run_dir(output_dir)
    output_path = run_dir / "smoke_posterior_samples.npz"
    np.savez(
        output_path,
        alpha=samples[:, 0].detach().cpu().numpy(),
        beta=samples[:, 1].detach().cpu().numpy(),
        delta=samples[:, 2].detach().cpu().numpy(),
        gamma=samples[:, 3].detach().cpu().numpy(),
    )

    print(f"Neural SBI smoke test completed with {num_simulations} simulations.")
    print(f"Saved test posterior samples to {output_path}")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a neural SBI smoke test.")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=200,
        help="Number of simulator calls to train the posterior estimator.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of posterior samples to draw.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Base directory for outputs.",
    )
    args = parser.parse_args()

    run_smoke_test(
        rng_seed=args.seed,
        num_simulations=args.num_simulations,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
