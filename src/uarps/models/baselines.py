"""Baseline B0: historical collision rate per cell and window of the week."""

from __future__ import annotations

import numpy as np
import pandas as pd


class HistoricalRateBaseline:
    """Predict the smoothed historical share of positive windows for each (cell, slot).

    The rate is shrunk towards the cell's overall rate, and the cell's rate towards the
    global rate, so that cells with few observations do not get extreme values::

        rate_cell      = (pos_cell + a * rate_global) / (n_cell + a)
        rate_cell_slot = (pos_cs   + a * rate_cell)   / (n_cs   + a)

    This mirrors common hotspot practice and is the naive reference model.
    """

    def __init__(
        self, cell_col: str = "cell", slot_col: str = "window_of_week", smoothing: float = 50.0
    ) -> None:
        self.cell_col = cell_col
        self.slot_col = slot_col
        self.smoothing = smoothing

    def fit(self, df: pd.DataFrame, y: pd.Series | np.ndarray) -> HistoricalRateBaseline:
        a = self.smoothing
        data = df[[self.cell_col, self.slot_col]].copy()
        data["y"] = np.asarray(y, dtype=float)

        self.global_rate_ = float(data["y"].mean())
        cell = data.groupby(self.cell_col)["y"].agg(["sum", "count"])
        self.cell_rate_ = (cell["sum"] + a * self.global_rate_) / (cell["count"] + a)

        cs = data.groupby([self.cell_col, self.slot_col])["y"].agg(["sum", "count"]).reset_index()
        prior = cs[self.cell_col].map(self.cell_rate_)
        cs["rate"] = (cs["sum"] + a * prior) / (cs["count"] + a)
        self.cell_slot_rate_ = cs.set_index([self.cell_col, self.slot_col])["rate"]
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_frame(df[[self.cell_col, self.slot_col]])
        rate = self.cell_slot_rate_.reindex(keys).to_numpy()
        fallback = df[self.cell_col].map(self.cell_rate_).to_numpy(dtype=float)
        rate = np.where(np.isnan(rate), fallback, rate)
        return np.where(np.isnan(rate), self.global_rate_, rate)
