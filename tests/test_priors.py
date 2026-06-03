from __future__ import annotations

import pytest

from predator_prey_sbi.priors import build_structure_aware_prior


def _obs() -> tuple[list[float], list[float]]:
    hare = [20.0, 22.0, 25.0, 21.0, 19.0]
    lynx = [10.0, 12.0, 11.0, 9.0, 8.0]
    return hare, lynx


def test_process_noise_correlation_requires_process_noise() -> None:
    hare, lynx = _obs()
    with pytest.raises(ValueError, match="requires include_process_noise=True"):
        build_structure_aware_prior(
            hare,
            lynx,
            include_process_noise=False,
            include_process_noise_correlation=True,
        )


def test_include_observation_ar1_adds_phi_parameters() -> None:
    hare, lynx = _obs()
    parameter_order, prior = build_structure_aware_prior(
        hare,
        lynx,
        include_observation_ar1=True,
    )
    assert "phi_h" in parameter_order
    assert "phi_l" in parameter_order
    assert prior["phi_h"] == [-0.95, 0.95]
    assert prior["phi_l"] == [-0.95, 0.95]
