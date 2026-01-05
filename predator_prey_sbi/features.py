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
    hare_peaks = _peak_rate(hare_log)
    lynx_peaks = _peak_rate(lynx_log)
    hare_period = _dominant_period(hare_log)
    lynx_period = _dominant_period(lynx_log)
    cross_corr = _corrcoef_safe(hare_log, lynx_log)
    cross_max_corr, cross_lag = _cross_corr_max_lag(hare_log, lynx_log, max_lag=10)

    return [
        hare_stats[0],
        hare_stats[1],
        hare_stats[2],
        hare_stats[3],
        hare_peaks,
        hare_period,
        lynx_stats[0],
        lynx_stats[1],
        lynx_stats[2],
        lynx_stats[3],
        lynx_peaks,
        lynx_period,
        cross_corr,
        cross_max_corr,
        cross_lag,
    ]


def _basic_stats(series: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(series))
    std = float(np.std(series))
    autocorr_1 = _autocorr(series, lag=1)
    autocorr_2 = _autocorr(series, lag=2)
    return mean, std, autocorr_1, autocorr_2


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


def _peak_rate(series: np.ndarray) -> float:
    if series.size < 3:
        return 0.0
    median = float(np.median(series))
    peaks = 0
    for idx in range(1, series.size - 1):
        if (
            series[idx] > median
            and series[idx] > series[idx - 1]
            and series[idx] > series[idx + 1]
        ):
            peaks += 1
    return float(peaks) / float(series.size)


def _dominant_period(series: np.ndarray) -> float:
    centered = series - np.mean(series)
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size <= 1:
        return float(series.size)
    spectrum[0] = 0.0
    idx = int(np.argmax(spectrum))
    if idx <= 0:
        return float(series.size)
    return float(series.size) / float(idx)


def _cross_corr_max_lag(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int,
) -> tuple[float, float]:
    if x.size != y.size or x.size == 0:
        return 0.0, 0.0

    best_corr = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_slice = x[:lag]
            y_slice = y[-lag:]
        elif lag > 0:
            x_slice = x[lag:]
            y_slice = y[:-lag]
        else:
            x_slice = x
            y_slice = y
        corr = _corrcoef_safe(x_slice, y_slice)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    lag_norm = float(best_lag) / float(x.size)
    return float(best_corr), lag_norm
