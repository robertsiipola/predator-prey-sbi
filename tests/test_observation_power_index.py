from __future__ import annotations

import numpy as np
import pytest

from predator_prey_sbi.simulator import simulate_lv


def test_observation_power_index_requires_reference() -> None:
    years = [1900.0, 1901.0, 1902.0]
    params = {"alpha": 0.0, "beta": 0.0, "delta": 0.0, "gamma": 0.0}
    with pytest.raises(ValueError, match="observation_reference is required"):
        simulate_lv(
            years=years,
            params=params,
            x0=(5.0, 10.0),
            dt=0.1,
            noise_scale=0.0,
            rng_seed=0,
            observation_power=(2.0, 2.0),
            observation_reference=None,
        )


def test_observation_power_index_applies_equilibrium_centered_mapping() -> None:
    years = [1900.0, 1901.0, 1902.0]
    params = {"alpha": 0.0, "beta": 0.0, "delta": 0.0, "gamma": 0.0}
    hare, lynx = simulate_lv(
        years=years,
        params=params,
        x0=(5.0, 10.0),
        dt=0.1,
        noise_scale=0.0,
        rng_seed=0,
        observation_power=(2.0, 2.0),
        observation_reference=(10.0, 20.0),
    )
    hare_arr = np.asarray(hare)
    lynx_arr = np.asarray(lynx)
    assert np.allclose(hare_arr, 10.0 * (5.0 / 10.0) ** 2)
    assert np.allclose(lynx_arr, 20.0 * (10.0 / 20.0) ** 2)
