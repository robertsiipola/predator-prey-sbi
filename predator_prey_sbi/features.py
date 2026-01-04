from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def summarize_series(hare: Iterable[float], lynx: Iterable[float]) -> list[float]:
    """Compute summary statistics used as inputs to the neural posterior estimator."""
    hare_arr = np.asarray(list(hare), dtype=float)
    lynx_arr = np.asarray(list(lynx), dtype=float)

    if hare_arr.size < 2 or lynx_arr.size < 2:
        raise ValueError("Hare and lynx series must have at least two observations")

    hare_log = np.log1p(np.clip(hare_arr, 0.0, None))
    lynx_log = np.log1p(np.clip(lynx_arr, 0.0, None))

    hare_stats = _basic_stats(hare_log)
    lynx_stats = _basic_stats(lynx_log)
    cross_corr = _corrcoef_safe(hare_log, lynx_log)

    return [
        hare_stats[0],
        hare_stats[1],
        hare_stats[2],
        lynx_stats[0],
        lynx_stats[1],
        lynx_stats[2],
        cross_corr,
    ]


def _basic_stats(series: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(series))
    std = float(np.std(series))
    autocorr = _autocorr(series, lag=1)
    return mean, std, autocorr


def _autocorr(series: np.ndarray, lag: int) -> float:
    if series.size <= lag:
        return 0.0
    x0 = series[:-lag]
    x1 = series[lag:]
    return _corrcoef_safe(x0, x1)


def _corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if math.isclose(float(np.std(x)), 0.0) or math.isclose(float(np.std(y)), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
