"""Leakage-safe historical accident features.

All functions here use only information from *before* each window, so they can
be computed at prediction time.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def decay_constant_from_half_life(half_life_windows: float) -> float:
    """Convert a half-life (in windows) into the decay constant kappa of Eq. 5."""
    if half_life_windows <= 0:
        raise ValueError("half_life_windows must be positive")
    return half_life_windows / math.log(2.0)


def decayed_intensity(counts: np.ndarray | pd.Series, kappa: float) -> np.ndarray:
    """Exponentially decayed collision intensity (Eq. 5 in the paper).

    ``h[t] = sum_{tau < t} exp(-(t - tau) / kappa) * N[tau]``

    The current window ``t`` is excluded, so the value is known before ``t`` starts.
    Computed in O(T) with the recursion ``h[t] = d * (h[t-1] + N[t-1])`` where
    ``d = exp(-1 / kappa)``.

    Parameters
    ----------
    counts:
        Collision counts per consecutive window for one cell, in time order, with no gaps.
    kappa:
        Decay constant measured in windows.
    """
    n = np.asarray(counts, dtype=float)
    d = math.exp(-1.0 / kappa)
    h = np.zeros_like(n)
    for t in range(1, len(n)):
        h[t] = d * (h[t - 1] + n[t - 1])
    return h


def past_window_sum(counts: np.ndarray | pd.Series, n_windows: int) -> np.ndarray:
    """Sum of counts over the previous ``n_windows`` windows, excluding the current one."""
    s = pd.Series(np.asarray(counts, dtype=float))
    return s.shift(1).rolling(n_windows, min_periods=1).sum().fillna(0.0).to_numpy()


def add_history_features(
    panel: pd.DataFrame,
    cell_col: str = "cell",
    time_col: str = "window_start",
    count_col: str = "n_collisions",
    window_hours: int = 3,
    history_days: tuple[int, ...] = (7, 30, 365),
    half_life_days: tuple[int, ...] = (30, 365),
) -> pd.DataFrame:
    """Add past-count and decayed-intensity features to a complete cell-window panel.

    ``panel`` must contain one row for every cell and every window (including windows
    with zero collisions); otherwise the window arithmetic below is not valid.
    """
    windows_per_day = 24 // window_hours
    out = panel.sort_values([cell_col, time_col]).copy()
    grouped = out.groupby(cell_col, sort=False)[count_col]

    for days in history_days:
        k = days * windows_per_day
        out[f"hist_count_{days}d"] = grouped.transform(lambda c, k=k: past_window_sum(c, k))

    for days in half_life_days:
        kappa = decay_constant_from_half_life(days * windows_per_day)
        out[f"hist_decay_{days}d"] = grouped.transform(
            lambda c, kappa=kappa: decayed_intensity(c, kappa)
        )

    return out
