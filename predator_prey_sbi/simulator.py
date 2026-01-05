from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def simulate_lv(
    years: list[float],
    params: dict[str, float],
    x0: tuple[float, float],
    dt: float,
    noise_scale: float | tuple[float, float],
    rng_seed: int | None,
) -> tuple[list[float], list[float]]:
    """Simulate prey/predator series with Lotka-Volterra ODE and log-normal observation noise."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if len(years) < 2:
        raise ValueError("years must contain at least two entries")

    years_array = np.asarray(years, dtype=float)
    if not np.all(np.diff(years_array) > 0):
        raise ValueError("years must be strictly increasing")

    t_eval = years_array - years_array[0]
    alpha = params["alpha"]
    beta = params["beta"]
    delta = params["delta"]
    gamma = params["gamma"]
    k = params.get("k")
    if k is not None and k <= 0:
        raise ValueError("k (carrying capacity) must be positive")

    def dynamics(_: float, state: np.ndarray) -> np.ndarray:
        prey, predator = state
        if k is None:
            d_prey = alpha * prey - beta * prey * predator
        else:
            d_prey = alpha * prey * (1.0 - prey / k) - beta * prey * predator
        d_predator = delta * prey * predator - gamma * predator
        return np.array([d_prey, d_predator], dtype=float)

    solver = solve_ivp(
        dynamics,
        (t_eval[0], t_eval[-1]),
        np.array([x0[0], x0[1]], dtype=float),
        t_eval=t_eval,
        max_step=dt,
        rtol=1e-6,
        atol=1e-8,
    )

    if not solver.success:
        raise RuntimeError(f"Simulation failed: {solver.message}")

    prey = solver.y[0]
    predator = solver.y[1]

    noise = _normalize_noise_scale(noise_scale)
    if noise is not None:
        rng = np.random.default_rng(rng_seed)
        prey = _apply_log_noise(prey, noise[0], rng)
        predator = _apply_log_noise(predator, noise[1], rng)

    return prey.tolist(), predator.tolist()


def _apply_log_noise(
    series: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    safe = np.clip(series, 1e-9, None)
    log_series = np.log(safe)
    noisy = log_series + rng.normal(0.0, noise_scale, size=log_series.shape)
    return np.exp(noisy)


def _normalize_noise_scale(
    noise_scale: float | tuple[float, float],
) -> tuple[float, float] | None:
    if isinstance(noise_scale, tuple):
        sigma_h, sigma_l = noise_scale
    else:
        sigma_h = sigma_l = float(noise_scale)
    if sigma_h < 0 or sigma_l < 0:
        raise ValueError("noise_scale must be non-negative")
    if sigma_h == 0 and sigma_l == 0:
        return None
    return sigma_h, sigma_l
