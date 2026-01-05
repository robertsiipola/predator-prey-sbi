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
        "prior_scheme": "structure_aware",
        "feature_mode": "embedding",
        "embedding": {
            "model": "nsf",
            "z_score_x": "structured",
            "hidden_features": 64,
            "num_transforms": 5,
            "num_bins": 10,
            "in_channels": 4,
            "hidden_channels": 16,
            "embedding_dim": 32,
            "kernel_size": 5,
        },
        "parameter_order": [
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
        "prior": {
            "log_T": [1.791759469228055, 2.772588722239781],
            "log_r": [-0.6931471805599453, 0.6931471805599453],
            "log_x_eq": [3.019692910998834, 4.405987272118725],
            "log_y_eq": [2.694289285852173, 4.080583646972063],
            "log_k_ratio": [0.04879016416943205, 2.0794415416798357],
            "eps_h0": [-0.2, 0.2],
            "eps_l0": [-0.2, 0.2],
            "log_sigma_h": [-2.995732273553991, -0.916290731874155],
            "log_sigma_l": [-2.995732273553991, -0.916290731874155],
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
