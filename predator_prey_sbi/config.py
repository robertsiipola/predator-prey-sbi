from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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
    "inference": {
        "num_simulations": 2000,
        "num_samples": 300,
        "num_workers": 1,
        "seed": 0,
        "noise_scale": 0.0,
        "sample_with": "rejection",
        "mcmc_method": "slice_np",
        "parameter_order": [
            "alpha",
            "gamma",
            "x_star",
            "y_star",
            "k",
            "sigma_h",
            "sigma_l",
            "eps_h0",
            "eps_l0",
        ],
        "prior": {
            "alpha": [0.2, 3.0],
            "gamma": [0.2, 3.0],
            "x_star": [5.0, 200.0],
            "y_star": [2.0, 100.0],
            "k": [150.0, 2000.0],
            "sigma_h": [0.05, 0.8],
            "sigma_l": [0.05, 0.8],
            "eps_h0": [-0.3, 0.3],
            "eps_l0": [-0.3, 0.3],
        },
    },
    "diagnostics": {
        "posterior_draws": 300,
        "seed": 0,
        "noise_scale": 0.0,
        "sbc": {
            "num_simulations": 500,
            "num_datasets": 20,
            "num_posterior_samples": 200,
            "num_workers": 1,
            "seed": 0,
            "sample_with": "rejection",
            "mcmc_method": "slice_np",
        },
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a mapping")
    return _deep_update(DEFAULT_CONFIG, loaded)
