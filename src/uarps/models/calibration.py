"""Probability correction after negative subsampling and isotonic calibration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression


def prior_correction(p_sampled: Sequence[float], beta: float) -> np.ndarray:
    """Map probabilities from a model trained on subsampled negatives back to the true ratio.

    Implements Eq. 16 of the paper: ``p = beta * p_s / (beta * p_s + 1 - p_s)``,
    where ``beta`` is the share of negatives that were kept during training.
    """
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    p = np.clip(np.asarray(p_sampled, dtype=float), 0.0, 1.0)
    return beta * p / (beta * p + 1.0 - p)


class RiskCalibrator:
    """Prior correction followed by isotonic regression, fitted on the validation year."""

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, p_sampled: Sequence[float], y_true: Sequence[int]) -> RiskCalibrator:
        corrected = prior_correction(p_sampled, self.beta)
        self._iso.fit(corrected, np.asarray(y_true, dtype=float))
        self.is_fitted = True
        return self

    def transform(self, p_sampled: Sequence[float]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("call fit() on validation predictions first")
        return self._iso.predict(prior_correction(p_sampled, self.beta))
