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
    "observation_operator": "point",
    "observation_substeps": 10,
    "observation_lag": 0.0,
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
        "process_noise_scale": 0.0,
        "sample_with": "rejection",
        "mcmc_method": "slice_np",
        "prior_scheme": "structure_aware",
        "k_parameterization": "k_ratio",
        "include_process_noise": False,
        "include_process_noise_correlation": False,
        "include_holling": False,
        "include_observation_scale": False,
        "include_observation_power": False,
        "include_observation_lag": False,
        "include_observation_ar1": False,
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
        "embedding_transform": "log1p",
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
            "log_T": [2.0794415416798357, 2.6390573296152584],
            "log_r": [-0.35667494393873245, 0.3364722366212129],
            "log_x_eq": [3.202014467792789, 4.182843720804516],
            "log_y_eq": [2.8766108426461274, 3.857440095657854],
            "log_k_ratio": [0.6931471805599453, 3.4011973816621555],
            "eps_h0": [-0.15, 0.15],
            "eps_l0": [-0.15, 0.15],
            "log_sigma_h": [-2.995732273553991, -1.2039728043259361],
            "log_sigma_l": [-2.995732273553991, -1.2039728043259361],
        },
    },
    "diagnostics": {
        "posterior_draws": 300,
        "seed": 0,
        "noise_scale": 0.0,
        "process_noise_scale": 0.0,
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
    config_path = Path(path).resolve()
    return _load_config_with_extends(config_path, seen=set())


def _load_config_with_extends(
    config_path: Path,
    seen: set[Path],
) -> dict[str, Any]:
    if config_path in seen:
        raise ValueError("Config extends cycle detected")
    seen.add(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a mapping")

    extends_value = loaded.get("extends")
    base = dict(DEFAULT_CONFIG)
    if extends_value is not None:
        if not isinstance(extends_value, str) or not extends_value.strip():
            raise ValueError("Config 'extends' must be a non-empty string path")
        base_path = (config_path.parent / extends_value).resolve()
        base = _load_config_with_extends(base_path, seen)

    merged = dict(loaded)
    merged.pop("extends", None)
    return _deep_update(base, merged)
