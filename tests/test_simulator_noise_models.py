from __future__ import annotations

import numpy as np
import pytest

from predator_prey_sbi.simulator import simulate_lv


def _years(n: int) -> list[float]:
    return [float(1900 + i) for i in range(n)]


def _acf_lag1(series: np.ndarray) -> float:
    centered = series - float(np.mean(series))
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denom)


def test_simulate_lv_rejects_invalid_process_noise_correlation() -> None:
    with pytest.raises(
        ValueError, match="process_noise_correlation must be in \\(-1, 1\\)"
    ):
        simulate_lv(
            years=_years(6),
            params={"alpha": 0.0, "beta": 0.0, "delta": 0.0, "gamma": 0.0},
            x0=(10.0, 10.0),
            dt=0.1,
            noise_scale=0.0,
            rng_seed=0,
            process_noise_scale=0.1,
            process_noise_correlation=1.0,
        )


def test_process_noise_correlation_controls_cross_species_shocks() -> None:
    years = _years(220)
    params = {"alpha": 0.0, "beta": 0.0, "delta": 0.0, "gamma": 0.0}
    hare_pos, lynx_pos = simulate_lv(
        years=years,
        params=params,
        x0=(10.0, 10.0),
        dt=0.1,
        noise_scale=0.0,
        rng_seed=123,
        process_noise_scale=0.25,
        process_noise_correlation=0.9,
        observation_operator="point",
    )
    hare_neg, lynx_neg = simulate_lv(
        years=years,
        params=params,
        x0=(10.0, 10.0),
        dt=0.1,
        noise_scale=0.0,
        rng_seed=123,
        process_noise_scale=0.25,
        process_noise_correlation=-0.9,
        observation_operator="point",
    )

    dh_pos = np.diff(np.log(np.asarray(hare_pos)))
    dl_pos = np.diff(np.log(np.asarray(lynx_pos)))
    dh_neg = np.diff(np.log(np.asarray(hare_neg)))
    dl_neg = np.diff(np.log(np.asarray(lynx_neg)))
    corr_pos = float(np.corrcoef(dh_pos, dl_pos)[0, 1])
    corr_neg = float(np.corrcoef(dh_neg, dl_neg)[0, 1])

    assert corr_pos > 0.7
    assert corr_neg < -0.7


def test_observation_ar1_increases_log_residual_autocorrelation() -> None:
    years = _years(220)
    params = {"alpha": 0.0, "beta": 0.0, "delta": 0.0, "gamma": 0.0}
    hare_iid, _ = simulate_lv(
        years=years,
        params=params,
        x0=(10.0, 10.0),
        dt=0.1,
        noise_scale=(0.2, 0.2),
        rng_seed=7,
        process_noise_scale=0.0,
        observation_operator="point",
        observation_ar1=(0.0, 0.0),
    )
    hare_ar1, _ = simulate_lv(
        years=years,
        params=params,
        x0=(10.0, 10.0),
        dt=0.1,
        noise_scale=(0.2, 0.2),
        rng_seed=7,
        process_noise_scale=0.0,
        observation_operator="point",
        observation_ar1=(0.85, 0.0),
    )
    resid_iid = np.log(np.asarray(hare_iid) / 10.0)
    resid_ar1 = np.log(np.asarray(hare_ar1) / 10.0)
    acf_iid = _acf_lag1(resid_iid)
    acf_ar1 = _acf_lag1(resid_ar1)

    assert acf_ar1 > 0.5
    assert acf_ar1 > acf_iid + 0.3
