from __future__ import annotations

import pytest

from predator_prey_sbi.config import load_config
from predator_prey_sbi.data import load_lynx_hare
from predator_prey_sbi.priors import build_structure_aware_prior


def test_relaxed_eq_obs_scale_config_builds_expected_prior() -> None:
    config = load_config("configs/experiments/base_seq_4k_relaxed_eq_obs_scale.yaml")
    _, hare, lynx = load_lynx_hare("data/LynxHare.txt")
    inference_cfg = config["inference"]

    parameter_order, prior = build_structure_aware_prior(
        hare,
        lynx,
        inference_cfg["prior"],
        k_parameterization=str(inference_cfg["k_parameterization"]),
        include_process_noise=bool(inference_cfg["include_process_noise"]),
        include_observation_scale=bool(inference_cfg["include_observation_scale"]),
    )

    assert inference_cfg["include_observation_scale"] is True
    assert parameter_order == [
        "log_T",
        "log_r",
        "log_x_eq",
        "log_y_eq",
        "log_k_ratio",
        "log_sigma_p",
        "log_c_h",
        "log_c_l",
        "eps_h0",
        "eps_l0",
        "log_sigma_h",
        "log_sigma_l",
    ]
    assert prior["log_x_eq"] == pytest.approx([2.3265457304388892, 5.09913445267867])
    assert prior["log_y_eq"] == pytest.approx([2.0011421052922276, 4.773730827532009])
    assert prior["log_c_h"] == pytest.approx([-1.6094379124341003, 1.6094379124341003])
    assert prior["log_c_l"] == pytest.approx([-1.6094379124341003, 1.6094379124341003])
