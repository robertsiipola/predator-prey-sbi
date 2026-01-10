from __future__ import annotations

import math
from typing import Any

import numpy as np


def build_structure_aware_prior(
    hare_obs: list[float],
    lynx_obs: list[float],
    overrides: dict[str, Any] | None = None,
    k_parameterization: str = "k_ratio",
    include_process_noise: bool = False,
    include_holling: bool = False,
    include_observation_scale: bool = False,
    include_observation_power: bool = False,
    include_observation_lag: bool = False,
) -> tuple[list[str], dict[str, list[float]]]:
    hare_arr = np.asarray(hare_obs, dtype=float)
    lynx_arr = np.asarray(lynx_obs, dtype=float)
    if hare_arr.size == 0 or lynx_arr.size == 0:
        raise ValueError(
            "Observed series must be non-empty to build structure-aware prior"
        )

    hare_med = float(np.median(hare_arr))
    lynx_med = float(np.median(lynx_arr))
    if hare_med <= 0 or lynx_med <= 0:
        raise ValueError("Observed series medians must be positive for log priors")

    def log_range(low: float, high: float) -> list[float]:
        if low <= 0 or high <= 0:
            raise ValueError("Log prior bounds must be positive")
        if low >= high:
            raise ValueError("Log prior bounds must have low < high")
        return [math.log(low), math.log(high)]

    defaults: dict[str, list[float]] = {
        "log_T": log_range(8.0, 14.0),
        "log_r": log_range(0.7, 1.4),
        "log_x_eq": log_range(0.6 * hare_med, 1.6 * hare_med),
        "log_y_eq": log_range(0.6 * lynx_med, 1.6 * lynx_med),
        "eps_h0": [-0.15, 0.15],
        "eps_l0": [-0.15, 0.15],
        "log_sigma_h": log_range(0.05, 0.3),
        "log_sigma_l": log_range(0.05, 0.3),
    }

    k_param = k_parameterization.strip().lower()
    if k_param == "k_ratio":
        defaults["log_k_ratio"] = log_range(2.0, 30.0)
        parameter_order = [
            "log_T",
            "log_r",
            "log_x_eq",
            "log_y_eq",
            "log_k_ratio",
            "eps_h0",
            "eps_l0",
            "log_sigma_h",
            "log_sigma_l",
        ]
    elif k_param == "tau_damp":
        defaults["log_tau_damp"] = log_range(20.0, 200.0)
        parameter_order = [
            "log_T",
            "log_r",
            "log_x_eq",
            "log_y_eq",
            "log_tau_damp",
            "eps_h0",
            "eps_l0",
            "log_sigma_h",
            "log_sigma_l",
        ]
    else:
        raise ValueError(
            "k_parameterization must be 'k_ratio' or 'tau_damp', got: "
            f"{k_parameterization!r}"
        )

    if include_process_noise:
        defaults["log_sigma_p"] = log_range(1e-3, 0.12)
        insert_at = parameter_order.index("eps_h0")
        parameter_order.insert(insert_at, "log_sigma_p")

    if include_holling:
        h_low = max(1e-4, 0.1 / hare_med)
        h_high = min(1.0, 10.0 / hare_med)
        if h_low >= h_high:
            raise ValueError("Invalid Holling-II prior bounds derived from data")
        defaults["log_h"] = log_range(h_low, h_high)
        insert_at = parameter_order.index("eps_h0")
        parameter_order.insert(insert_at, "log_h")

    if include_observation_scale:
        defaults["log_c_h"] = log_range(0.2, 5.0)
        defaults["log_c_l"] = log_range(0.2, 5.0)
        insert_at = parameter_order.index("eps_h0")
        parameter_order.insert(insert_at, "log_c_l")
        parameter_order.insert(insert_at, "log_c_h")

    if include_observation_power:
        defaults["p"] = [0.7, 1.3]
        insert_at = parameter_order.index("eps_h0")
        parameter_order.insert(insert_at, "p")

    if include_observation_lag:
        defaults["obs_lag"] = [0.0, 1.0]
        insert_at = parameter_order.index("eps_h0")
        parameter_order.insert(insert_at, "obs_lag")

    if overrides:
        for key, value in overrides.items():
            if key in defaults:
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(f"Override for {key} must be a two-element list")
                defaults[key] = [float(value[0]), float(value[1])]

    return parameter_order, defaults
