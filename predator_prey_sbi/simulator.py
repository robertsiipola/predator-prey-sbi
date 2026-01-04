from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def simulate_lv(
    years: list[float],
    params: dict[str, float],
    x0: tuple[float, float],
    dt: float,
    noise_scale: float,
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

    def dynamics(_: float, state: np.ndarray) -> np.ndarray:
        prey, predator = state
        d_prey = alpha * prey - beta * prey * predator
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

    if noise_scale > 0:
        rng = np.random.default_rng(rng_seed)
        prey = _apply_log_noise(prey, noise_scale, rng)
        predator = _apply_log_noise(predator, noise_scale, rng)

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
