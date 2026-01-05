from __future__ import annotations

import math
from typing import Any

import numpy as np


def build_structure_aware_prior(
    hare_obs: list[float],
    lynx_obs: list[float],
    overrides: dict[str, Any] | None = None,
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
        "log_T": log_range(6.0, 16.0),
        "log_r": log_range(0.5, 2.0),
        "log_x_eq": log_range(0.5 * hare_med, 2.0 * hare_med),
        "log_y_eq": log_range(0.5 * lynx_med, 2.0 * lynx_med),
        "log_k_ratio": log_range(1.05, 8.0),
        "eps_h0": [-0.2, 0.2],
        "eps_l0": [-0.2, 0.2],
        "log_sigma_h": log_range(0.05, 0.4),
        "log_sigma_l": log_range(0.05, 0.4),
    }

    if overrides:
        for key, value in overrides.items():
            if key in defaults:
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(f"Override for {key} must be a two-element list")
                defaults[key] = [float(value[0]), float(value[1])]

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

    return parameter_order, defaults
