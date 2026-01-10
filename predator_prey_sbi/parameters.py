from __future__ import annotations

import math
from typing import Mapping


def resolve_lv_params(values: Mapping[str, float]) -> dict[str, float]:
    if {"log_T", "log_r", "log_x_eq", "log_y_eq"}.issubset(values) and (
        "log_k_ratio" in values or "log_tau_damp" in values
    ):
        log_t = float(values["log_T"])
        log_r = float(values["log_r"])
        log_x_eq = float(values["log_x_eq"])
        log_y_eq = float(values["log_y_eq"])

        t = math.exp(log_t)
        r = math.exp(log_r)
        x_eq = math.exp(log_x_eq)
        y_eq = math.exp(log_y_eq)

        if t <= 0 or r <= 0 or x_eq <= 0 or y_eq <= 0:
            raise ValueError("Structure-aware parameters must be positive")

        if "log_tau_damp" in values:
            tau_damp = math.exp(float(values["log_tau_damp"]))
            if tau_damp <= 0:
                raise ValueError("tau_damp must be positive")
            k_ratio = tau_damp * math.pi * r / t
        else:
            k_ratio = math.exp(float(values["log_k_ratio"]))
        if k_ratio <= 1.0:
            raise ValueError("k_ratio must be greater than 1 to keep K > x_eq")

        k = x_eq * k_ratio
        s = 2.0 * math.pi / t
        alpha = s * r
        gamma = s / r
        delta = gamma / x_eq
        coexistence = 1.0 - x_eq / k
        if coexistence <= 0:
            raise ValueError("Coexistence factor must be positive")
        beta = alpha * coexistence / y_eq
        return {
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "gamma": gamma,
            "k": k,
        }

    k = None
    if "k" in values:
        k = float(values["k"])
        if k <= 0:
            raise ValueError("k (carrying capacity) must be positive")
    if {"x_star", "y_star"}.issubset(values):
        for key in ("alpha", "gamma", "x_star", "y_star"):
            if key not in values:
                raise ValueError(f"Missing required parameter: {key}")
        alpha = float(values["alpha"])
        gamma = float(values["gamma"])
        x_star = float(values["x_star"])
        y_star = float(values["y_star"])
        if x_star <= 0 or y_star <= 0:
            raise ValueError("x_star and y_star must be positive")
        beta = alpha / y_star
        delta = gamma / x_star
    else:
        for key in ("alpha", "beta", "delta", "gamma"):
            if key not in values:
                raise ValueError(f"Missing required parameter: {key}")
        alpha = float(values["alpha"])
        beta = float(values["beta"])
        delta = float(values["delta"])
        gamma = float(values["gamma"])

    params = {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "gamma": gamma,
    }
    if k is not None:
        params["k"] = k
    if "log_h" in values:
        h = math.exp(float(values["log_h"]))
        if h <= 0:
            raise ValueError("h must be positive")
        params["h"] = h
    elif "h" in values:
        h = float(values["h"])
        if h <= 0:
            raise ValueError("h must be positive")
        params["h"] = h
    return params


def resolve_noise_scales(
    values: Mapping[str, float],
    default_noise: float | tuple[float, float],
) -> float | tuple[float, float]:
    if "log_sigma_h" in values or "log_sigma_l" in values:
        if "log_sigma_h" not in values or "log_sigma_l" not in values:
            raise ValueError("Both log_sigma_h and log_sigma_l are required")
        sigma_h = math.exp(float(values["log_sigma_h"]))
        sigma_l = math.exp(float(values["log_sigma_l"]))
        if sigma_h < 0 or sigma_l < 0:
            raise ValueError("log_sigma_h/log_sigma_l must map to non-negative values")
        return sigma_h, sigma_l
    if "sigma_h" in values or "sigma_l" in values:
        if "sigma_h" not in values or "sigma_l" not in values:
            raise ValueError("Both sigma_h and sigma_l are required")
        sigma_h = float(values["sigma_h"])
        sigma_l = float(values["sigma_l"])
        if sigma_h < 0 or sigma_l < 0:
            raise ValueError("sigma_h and sigma_l must be non-negative")
        return sigma_h, sigma_l
    return default_noise


def resolve_process_noise_scale(
    values: Mapping[str, float],
    default_scale: float,
) -> float:
    if "log_sigma_p" in values:
        sigma_p = math.exp(float(values["log_sigma_p"]))
        if sigma_p < 0:
            raise ValueError("log_sigma_p must map to non-negative values")
        return sigma_p
    if "sigma_p" in values:
        sigma_p = float(values["sigma_p"])
        if sigma_p < 0:
            raise ValueError("sigma_p must be non-negative")
        return sigma_p
    return float(default_scale)


def resolve_observation_scales(
    values: Mapping[str, float],
    default_scales: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, float]:
    if "log_c_h" in values or "log_c_l" in values:
        if "log_c_h" not in values or "log_c_l" not in values:
            raise ValueError("Both log_c_h and log_c_l are required")
        c_h = math.exp(float(values["log_c_h"]))
        c_l = math.exp(float(values["log_c_l"]))
        if c_h <= 0 or c_l <= 0:
            raise ValueError("log_c_h/log_c_l must map to positive values")
        return c_h, c_l
    if "c_h" in values or "c_l" in values:
        if "c_h" not in values or "c_l" not in values:
            raise ValueError("Both c_h and c_l are required")
        c_h = float(values["c_h"])
        c_l = float(values["c_l"])
        if c_h <= 0 or c_l <= 0:
            raise ValueError("c_h and c_l must be positive")
        return c_h, c_l
    return default_scales


def resolve_observation_power(
    values: Mapping[str, float],
    default_power: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, float]:
    if "p" in values:
        p = float(values["p"])
        if not math.isfinite(p):
            raise ValueError("p must be finite")
        if p <= 0:
            raise ValueError("p must be positive")
        return p, p
    if "p_h" in values or "p_l" in values:
        if "p_h" not in values or "p_l" not in values:
            raise ValueError("Both p_h and p_l are required")
        p_h = float(values["p_h"])
        p_l = float(values["p_l"])
        if not (math.isfinite(p_h) and math.isfinite(p_l)):
            raise ValueError("p_h/p_l must be finite")
        if p_h <= 0 or p_l <= 0:
            raise ValueError("p_h/p_l must be positive")
        return p_h, p_l
    return default_power


def resolve_observation_lag(
    values: Mapping[str, float],
    default_lag: float = 0.0,
) -> float:
    if "obs_lag" in values:
        lag = float(values["obs_lag"])
        if not math.isfinite(lag):
            raise ValueError("obs_lag must be finite")
        if lag < 0:
            raise ValueError("obs_lag must be non-negative")
        return lag
    if "tau_obs" in values:
        lag = float(values["tau_obs"])
        if not math.isfinite(lag):
            raise ValueError("tau_obs must be finite")
        if lag < 0:
            raise ValueError("tau_obs must be non-negative")
        return lag
    return float(default_lag)


def resolve_initial_conditions(
    values: Mapping[str, float],
    base_x0: tuple[float, float],
) -> tuple[float, float]:
    if "eps_h0" in values or "eps_l0" in values:
        if "eps_h0" not in values or "eps_l0" not in values:
            raise ValueError("Both eps_h0 and eps_l0 are required")
        hare0 = base_x0[0] * math.exp(float(values["eps_h0"]))
        lynx0 = base_x0[1] * math.exp(float(values["eps_l0"]))
        return hare0, lynx0
    if "hare0" in values and "lynx0" in values:
        return float(values["hare0"]), float(values["lynx0"])
    return base_x0
