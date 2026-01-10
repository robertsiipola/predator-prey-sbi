from __future__ import annotations

import math

import pytest

from predator_prey_sbi.parameters import resolve_lv_params, resolve_observation_lag


def test_resolve_observation_lag_prefers_obs_lag() -> None:
    assert resolve_observation_lag({"obs_lag": 0.25, "tau_obs": 0.5}) == pytest.approx(
        0.25
    )


def test_resolve_observation_lag_rejects_negative() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        resolve_observation_lag({"obs_lag": -0.1})


def test_resolve_lv_params_tau_damp_parameterization() -> None:
    values = {
        "log_T": math.log(10.0),
        "log_r": math.log(1.0),
        "log_x_eq": math.log(2.0),
        "log_y_eq": math.log(3.0),
        "log_tau_damp": math.log(100.0),
    }
    params = resolve_lv_params(values)

    t = 10.0
    r = 1.0
    x_eq = 2.0
    y_eq = 3.0
    tau_damp = 100.0
    k_ratio = tau_damp * math.pi * r / t
    k = x_eq * k_ratio
    s = 2.0 * math.pi / t
    alpha = s * r
    gamma = s / r
    delta = gamma / x_eq
    coexistence = 1.0 - x_eq / k
    beta = alpha * coexistence / y_eq

    assert params["k"] == pytest.approx(k)
    assert params["alpha"] == pytest.approx(alpha)
    assert params["gamma"] == pytest.approx(gamma)
    assert params["delta"] == pytest.approx(delta)
    assert params["beta"] == pytest.approx(beta)


def test_resolve_lv_params_k_ratio_must_exceed_one() -> None:
    values = {
        "log_T": math.log(10.0),
        "log_r": math.log(1.0),
        "log_x_eq": math.log(2.0),
        "log_y_eq": math.log(3.0),
        "log_k_ratio": math.log(0.9),
    }
    with pytest.raises(ValueError, match="k_ratio must be greater than 1"):
        resolve_lv_params(values)
