from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

from predator_prey_sbi.data import load_lynx_hare
from predator_prey_sbi.simulator import simulate_lv


DEFAULT_CONFIG: dict[str, Any] = {
    "data_path": "data/LynxHare.txt",
    "params": {
        "alpha": 1.5,
        "beta": 0.9,
        "delta": 0.75,
        "gamma": 1.3,
    },
    "noise_scale": 0.1,
    "dt": 0.1,
    "use_observed_initial": True,
    "x0": [10.0, 10.0],
    "plot": {
        "show_observed": True,
    },
    "output_dir": "runs",
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a mapping")
    return _deep_update(DEFAULT_CONFIG, loaded)


def _make_run_dir(base_dir: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _plot_series(
    years: list[float],
    observed: tuple[list[float], list[float]],
    simulated: tuple[list[float], list[float]],
    output_path: Path,
    show_observed: bool,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    hare_obs, lynx_obs = observed
    hare_sim, lynx_sim = simulated

    axes[0].plot(years, hare_sim, label="Simulated hare", color="tab:blue")
    axes[1].plot(years, lynx_sim, label="Simulated lynx", color="tab:orange")

    if show_observed:
        axes[0].plot(
            years, hare_obs, label="Observed hare", color="tab:blue", linestyle="--"
        )
        axes[1].plot(
            years, lynx_obs, label="Observed lynx", color="tab:orange", linestyle="--"
        )

    axes[0].set_ylabel("Hare")
    axes[1].set_ylabel("Lynx")
    axes[1].set_xlabel("Year")

    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_simulation(config_path: str) -> Path:
    config = _load_config(config_path)

    data_path = str(config["data_path"])
    years, hare_obs, lynx_obs = load_lynx_hare(data_path)

    if config.get("use_observed_initial", True):
        x0 = (hare_obs[0], lynx_obs[0])
    else:
        x0_values = config.get("x0", [10.0, 10.0])
        if len(x0_values) != 2:
            raise ValueError("x0 must contain two values")
        x0 = (float(x0_values[0]), float(x0_values[1]))

    params = config["params"]
    noise_scale = float(config["noise_scale"])
    dt = float(config["dt"])
    rng_seed = config.get("rng_seed")

    hare_sim, lynx_sim = simulate_lv(
        years=years,
        params=params,
        x0=x0,
        dt=dt,
        noise_scale=noise_scale,
        rng_seed=rng_seed,
    )

    run_dir = _make_run_dir(str(config.get("output_dir", "runs")))
    plot_path = run_dir / "simulated_trajectories.png"
    show_observed = bool(config.get("plot", {}).get("show_observed", True))

    _plot_series(
        years=years,
        observed=(hare_obs, lynx_obs),
        simulated=(hare_sim, lynx_sim),
        output_path=plot_path,
        show_observed=show_observed,
    )

    print(f"Loaded {data_path} with {len(years)} yearly observations.")
    print(
        "Simulated {n} steps using Lotka-Volterra parameters: "
        "alpha={alpha}, beta={beta}, delta={delta}, gamma={gamma}".format(
            n=len(years),
            alpha=params["alpha"],
            beta=params["beta"],
            delta=params["delta"],
            gamma=params["gamma"],
        )
    )
    print(f"Wrote plots to {plot_path}")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate predator-prey trajectories.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()
    run_simulation(args.config)


if __name__ == "__main__":
    main()
