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
    process_noise_scale: float = 0.0,
    process_noise_correlation: float = 0.0,
    observation_scale: tuple[float, float] = (1.0, 1.0),
    observation_power: tuple[float, float] = (1.0, 1.0),
    observation_reference: tuple[float, float] | None = None,
    observation_operator: str = "point",
    observation_substeps: int = 10,
    observation_lag: float = 0.0,
    observation_ar1: tuple[float, float] = (0.0, 0.0),
) -> tuple[list[float], list[float]]:
    """Simulate prey/predator series with Lotka-Volterra ODE and log-normal observation noise."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if len(years) < 2:
        raise ValueError("years must contain at least two entries")
    if observation_substeps <= 0:
        raise ValueError("observation_substeps must be positive")
    if not np.isfinite(observation_lag):
        raise ValueError("observation_lag must be finite")
    if observation_lag < 0:
        raise ValueError("observation_lag must be non-negative")
    if process_noise_scale < 0:
        raise ValueError("process_noise_scale must be non-negative")
    if not np.isfinite(process_noise_correlation):
        raise ValueError("process_noise_correlation must be finite")
    if abs(process_noise_correlation) >= 1.0:
        raise ValueError("process_noise_correlation must be in (-1, 1)")
    if observation_scale[0] <= 0 or observation_scale[1] <= 0:
        raise ValueError("observation_scale values must be positive")
    if observation_power[0] <= 0 or observation_power[1] <= 0:
        raise ValueError("observation_power values must be positive")
    if observation_reference is not None and (
        observation_reference[0] <= 0 or observation_reference[1] <= 0
    ):
        raise ValueError("observation_reference values must be positive")
    if not (np.isfinite(observation_ar1[0]) and np.isfinite(observation_ar1[1])):
        raise ValueError("observation_ar1 values must be finite")
    if abs(observation_ar1[0]) >= 1.0 or abs(observation_ar1[1]) >= 1.0:
        raise ValueError("observation_ar1 values must be in (-1, 1)")

    years_array = np.asarray(years, dtype=float)
    if not np.all(np.diff(years_array) > 0):
        raise ValueError("years must be strictly increasing")

    t_eval = years_array - years_array[0]
    alpha = params["alpha"]
    beta = params["beta"]
    delta = params["delta"]
    gamma = params["gamma"]
    k = params.get("k")
    h = params.get("h")
    if k is not None and k <= 0:
        raise ValueError("k (carrying capacity) must be positive")
    if h is not None and h <= 0:
        raise ValueError("h (handling time) must be positive")

    def dynamics(_: float, state: np.ndarray) -> np.ndarray:
        prey, predator = state
        if h is None:
            predation = prey * predator
        else:
            predation = (prey * predator) / (1.0 + h * prey)
        prey_loss = beta * predation
        predator_gain = delta * predation
        if k is None:
            d_prey = alpha * prey - prey_loss
        else:
            d_prey = alpha * prey * (1.0 - prey / k) - prey_loss
        d_predator = predator_gain - gamma * predator
        return np.array([d_prey, d_predator], dtype=float)

    op = observation_operator.strip().lower()
    if process_noise_scale > 0:
        rng = np.random.default_rng(rng_seed)
        prey, predator = _simulate_with_process_noise(
            t_eval=t_eval,
            x0=x0,
            dt=dt,
            process_noise_scale=process_noise_scale,
            rng=rng,
            alpha=alpha,
            beta=beta,
            delta=delta,
            gamma=gamma,
            k=k,
            h=h,
            observation_operator=op,
            observation_substeps=observation_substeps,
            observation_lag=observation_lag,
            process_noise_correlation=process_noise_correlation,
        )
        prey, predator = _apply_power_index(
            prey,
            predator,
            observation_power=observation_power,
            observation_reference=observation_reference,
        )
        prey = prey * float(observation_scale[0])
        predator = predator * float(observation_scale[1])
        noise = _normalize_noise_scale(noise_scale)
        if noise is not None:
            prey = _apply_observation_noise(prey, noise[0], observation_ar1[0], rng)
            predator = _apply_observation_noise(
                predator, noise[1], observation_ar1[1], rng
            )
        return prey.tolist(), predator.tolist()

    t_start = float(t_eval[0])
    if op in {"point", "points"}:
        t_end = float(t_eval[-1] + observation_lag)
        sample_times = t_eval + observation_lag
        needs_dense = False
    elif op in {"midpoint", "mid_year", "midyear"}:
        t_end = float(t_eval[-1] + 0.5 + observation_lag)
        sample_times = t_eval + 0.5 + observation_lag
        needs_dense = False
    elif op in {"annual_mean", "annual_average", "year_mean", "year_average"}:
        diffs = np.diff(t_eval)
        step = float(diffs[0])
        if not np.allclose(diffs, step, rtol=0.0, atol=1e-9):
            raise ValueError("annual_mean observation requires evenly spaced years")
        if observation_lag >= step:
            raise ValueError("observation_lag must be less than the year step")
        t_end = float(t_eval[-1] + step + observation_lag)
        sample_times = None
        needs_dense = True
    else:
        raise ValueError(
            "observation_operator must be one of: point, midpoint, annual_mean; got: "
            f"{observation_operator!r}"
        )

    solver = solve_ivp(
        dynamics,
        (t_start, t_end),
        np.array([x0[0], x0[1]], dtype=float),
        t_eval=sample_times,
        max_step=dt,
        rtol=1e-6,
        atol=1e-8,
        dense_output=needs_dense,
    )

    if not solver.success:
        raise RuntimeError(f"Simulation failed: {solver.message}")

    if op in {"annual_mean", "annual_average", "year_mean", "year_average"}:
        if solver.sol is None:
            raise RuntimeError("Dense output required but solver.sol is missing")
        diffs = np.diff(t_eval)
        step = float(diffs[0])
        offsets = (np.arange(observation_substeps, dtype=float) + 0.5) / float(
            observation_substeps
        )
        sample_grid = offsets * step
        prey_vals = np.empty(t_eval.shape[0], dtype=float)
        predator_vals = np.empty(t_eval.shape[0], dtype=float)
        for idx, t0 in enumerate(t_eval):
            times = t0 + observation_lag + sample_grid
            values = solver.sol(times)
            prey_vals[idx] = float(np.mean(values[0]))
            predator_vals[idx] = float(np.mean(values[1]))
        prey = prey_vals
        predator = predator_vals
    else:
        prey = solver.y[0]
        predator = solver.y[1]

    prey, predator = _apply_power_index(
        prey,
        predator,
        observation_power=observation_power,
        observation_reference=observation_reference,
    )
    prey = prey * float(observation_scale[0])
    predator = predator * float(observation_scale[1])

    noise = _normalize_noise_scale(noise_scale)
    if noise is not None:
        rng = np.random.default_rng(rng_seed)
        prey = _apply_observation_noise(prey, noise[0], observation_ar1[0], rng)
        predator = _apply_observation_noise(predator, noise[1], observation_ar1[1], rng)

    return prey.tolist(), predator.tolist()


def _apply_power_index(
    prey: np.ndarray,
    predator: np.ndarray,
    *,
    observation_power: tuple[float, float],
    observation_reference: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    p_h, p_l = float(observation_power[0]), float(observation_power[1])
    if p_h == 1.0 and p_l == 1.0:
        return prey, predator
    if observation_reference is None:
        raise ValueError(
            "observation_reference is required when observation_power != (1, 1)"
        )
    ref_h, ref_l = float(observation_reference[0]), float(observation_reference[1])
    if ref_h <= 0 or ref_l <= 0:
        raise ValueError("observation_reference values must be positive")
    safe_prey = np.clip(prey, 1e-9, None)
    safe_predator = np.clip(predator, 1e-9, None)
    prey_out = ref_h * np.exp(p_h * (np.log(safe_prey) - np.log(ref_h)))
    pred_out = ref_l * np.exp(p_l * (np.log(safe_predator) - np.log(ref_l)))
    return prey_out, pred_out


def _simulate_with_process_noise(
    t_eval: np.ndarray,
    x0: tuple[float, float],
    dt: float,
    process_noise_scale: float,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    k: float | None,
    h: float | None,
    observation_operator: str,
    observation_substeps: int,
    observation_lag: float,
    process_noise_correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    diffs = np.diff(t_eval)
    if diffs.size == 0:
        raise ValueError("years must contain at least two entries")

    def deriv(prey: float, predator: float) -> tuple[float, float]:
        if h is None:
            predation = prey * predator
        else:
            predation = (prey * predator) / (1.0 + h * prey)
        prey_loss = beta * predation
        predator_gain = delta * predation
        if k is None:
            d_prey = alpha * prey - prey_loss
        else:
            d_prey = alpha * prey * (1.0 - prey / k) - prey_loss
        d_pred = predator_gain - gamma * predator
        return d_prey, d_pred

    def integrate(
        t: float,
        prey: float,
        predator: float,
        t_target: float,
    ) -> tuple[float, float, float]:
        while t < t_target:
            h = min(dt, t_target - t)
            k1x, k1y = deriv(prey, predator)
            k2x, k2y = deriv(prey + 0.5 * h * k1x, predator + 0.5 * h * k1y)
            k3x, k3y = deriv(prey + 0.5 * h * k2x, predator + 0.5 * h * k2y)
            k4x, k4y = deriv(prey + h * k3x, predator + h * k3y)
            prey = prey + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            predator = predator + (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            t = t + h
            if not (np.isfinite(prey) and np.isfinite(predator)):
                raise RuntimeError("Simulation diverged (non-finite state)")
        return t, prey, predator

    op = observation_operator
    if op in {"midpoint", "mid_year", "midyear", "annual_mean", "annual_average"}:
        step = float(diffs[0])
        if not np.allclose(diffs, step, rtol=0.0, atol=1e-9):
            raise ValueError(
                "process noise simulation requires evenly spaced years for midpoint/annual_mean"
            )
        if observation_lag >= step:
            raise ValueError("observation_lag must be less than the year step")

    n = t_eval.shape[0]
    prey_out = np.empty(n, dtype=float)
    predator_out = np.empty(n, dtype=float)

    t = float(t_eval[0])
    prey = float(x0[0])
    predator = float(x0[1])

    if op in {"point", "points"}:
        for idx in range(n):
            t_obs = float(t_eval[idx] + observation_lag)
            t, prey, predator = integrate(t, prey, predator, t_obs)
            prey_out[idx] = prey
            predator_out[idx] = predator
            if idx == n - 1:
                break
            t_boundary = float(t_eval[idx + 1] + observation_lag)
            dt_year = float(t_boundary - t)
            t, prey, predator = integrate(t, prey, predator, t_boundary)
            scale = process_noise_scale * float(np.sqrt(max(dt_year, 0.0)))
            shock_h, shock_l = _draw_process_shocks(
                scale=scale,
                correlation=process_noise_correlation,
                rng=rng,
            )
            prey = max(prey, 1e-9) * float(np.exp(shock_h))
            predator = max(predator, 1e-9) * float(np.exp(shock_l))
        return prey_out, predator_out

    if op in {"midpoint", "mid_year", "midyear"}:
        step = float(diffs[0])
        for idx in range(n):
            mid = float(t_eval[idx] + observation_lag + 0.5 * step)
            t, prey, predator = integrate(t, prey, predator, mid)
            prey_out[idx] = prey
            predator_out[idx] = predator
            if idx == n - 1:
                break
            boundary = float(t_eval[idx + 1] + observation_lag)
            t, prey, predator = integrate(t, prey, predator, boundary)
            scale = process_noise_scale * float(np.sqrt(max(step, 0.0)))
            shock_h, shock_l = _draw_process_shocks(
                scale=scale,
                correlation=process_noise_correlation,
                rng=rng,
            )
            prey = max(prey, 1e-9) * float(np.exp(shock_h))
            predator = max(predator, 1e-9) * float(np.exp(shock_l))
        return prey_out, predator_out

    if op in {"annual_mean", "annual_average", "year_mean", "year_average"}:
        step = float(diffs[0])
        offsets = (np.arange(observation_substeps, dtype=float) + 0.5) / float(
            observation_substeps
        )
        for idx in range(n):
            start = float(t_eval[idx] + observation_lag)
            if t < start:
                t, prey, predator = integrate(t, prey, predator, start)
            samples_p = []
            samples_q = []
            for frac in offsets:
                ts = start + float(frac) * step
                t, prey, predator = integrate(t, prey, predator, ts)
                samples_p.append(prey)
                samples_q.append(predator)
            prey_out[idx] = float(np.mean(samples_p))
            predator_out[idx] = float(np.mean(samples_q))

            end = start + step
            t, prey, predator = integrate(t, prey, predator, end)
            if idx == n - 1:
                break
            scale = process_noise_scale * float(np.sqrt(max(step, 0.0)))
            shock_h, shock_l = _draw_process_shocks(
                scale=scale,
                correlation=process_noise_correlation,
                rng=rng,
            )
            prey = max(prey, 1e-9) * float(np.exp(shock_h))
            predator = max(predator, 1e-9) * float(np.exp(shock_l))
        return prey_out, predator_out

    raise ValueError(
        "observation_operator must be one of: point, midpoint, annual_mean; got: "
        f"{observation_operator!r}"
    )


def _apply_log_noise(
    series: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    safe = np.clip(series, 1e-9, None)
    log_series = np.log(safe)
    noisy = log_series + rng.normal(0.0, noise_scale, size=log_series.shape)
    return np.exp(noisy)


def _apply_observation_noise(
    series: np.ndarray,
    noise_scale: float,
    ar1: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_scale == 0.0:
        return np.asarray(series, dtype=float)
    if ar1 == 0.0:
        return _apply_log_noise(series, noise_scale, rng)
    safe = np.clip(series, 1e-9, None)
    log_series = np.log(safe)
    eps = np.empty(log_series.shape, dtype=float)
    init_sd = noise_scale / float(np.sqrt(max(1e-12, 1.0 - ar1**2)))
    eps_prev = float(rng.normal(0.0, init_sd))
    for idx in range(log_series.size):
        innov = float(rng.normal(0.0, noise_scale))
        eps_prev = ar1 * eps_prev + innov
        eps[idx] = eps_prev
    return np.exp(log_series + eps)


def _draw_process_shocks(
    *,
    scale: float,
    correlation: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if scale <= 0:
        return 0.0, 0.0
    if abs(correlation) < 1e-12:
        return float(rng.normal(0.0, scale)), float(rng.normal(0.0, scale))
    cov = np.array([[1.0, correlation], [correlation, 1.0]], dtype=float)
    pair = rng.multivariate_normal(mean=np.zeros(2, dtype=float), cov=cov)
    return float(pair[0] * scale), float(pair[1] * scale)


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
