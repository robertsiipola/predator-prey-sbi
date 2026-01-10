from __future__ import annotations

import numpy as np
import pytest

from predator_prey_sbi.simulator import simulate_lv


def _years(n: int) -> list[float]:
    return [float(1900 + i) for i in range(n)]


def test_simulate_lv_observation_lag_finite_required() -> None:
    years = _years(5)
    params = {"alpha": 1.0, "beta": 0.02, "delta": 0.01, "gamma": 1.0}
    with pytest.raises(ValueError, match="observation_lag must be finite"):
        simulate_lv(
            years=years,
            params=params,
            x0=(10.0, 10.0),
            dt=0.1,
            noise_scale=0.0,
            rng_seed=0,
            observation_lag=float("nan"),
        )


def test_simulate_lv_negative_observation_lag_supported_and_reproducible() -> None:
    years = _years(8)
    params = {"alpha": 1.0, "beta": 0.02, "delta": 0.01, "gamma": 1.0}
    with pytest.raises(ValueError, match="observation_lag must be non-negative"):
        simulate_lv(
            years=years,
            params=params,
            x0=(10.0, 10.0),
            dt=0.1,
            noise_scale=0.0,
            rng_seed=0,
            observation_operator="annual_mean",
            observation_substeps=5,
            observation_lag=-0.25,
        )


def test_simulate_lv_observation_lag_changes_latent_sample_points() -> None:
    years = _years(10)
    params = {"alpha": 1.0, "beta": 0.02, "delta": 0.01, "gamma": 1.0}
    hare_0, lynx_0 = simulate_lv(
        years=years,
        params=params,
        x0=(10.0, 10.0),
        dt=0.05,
        noise_scale=0.0,
        rng_seed=0,
        observation_operator="midpoint",
        observation_lag=0.0,
    )
    hare_shift, lynx_shift = simulate_lv(
        years=years,
        params=params,
        x0=(10.0, 10.0),
        dt=0.05,
        noise_scale=0.0,
        rng_seed=0,
        observation_operator="midpoint",
        observation_lag=0.25,
    )
    assert not np.allclose(hare_0, hare_shift)
    assert not np.allclose(lynx_0, lynx_shift)
