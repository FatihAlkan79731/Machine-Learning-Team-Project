"""Chronological train / validation / test splits and expanding-window backtests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitMasks:
    """Boolean row masks for one chronological split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _year_mask(years: pd.Series, selected: Iterable[int]) -> np.ndarray:
    return years.isin(list(selected)).to_numpy()


def temporal_split(
    df: pd.DataFrame,
    time_col: str,
    train_years: Sequence[int],
    validation_years: Sequence[int],
    test_years: Sequence[int],
) -> SplitMasks:
    """Split rows by calendar year. Raises if the year groups overlap or are out of order."""
    groups = [set(train_years), set(validation_years), set(test_years)]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
        raise ValueError("train, validation and test years must not overlap")
    if validation_years and max(train_years) >= min(validation_years):
        raise ValueError("validation years must come after all training years")
    if test_years and validation_years and max(validation_years) >= min(test_years):
        raise ValueError("test years must come after all validation years")

    years = pd.to_datetime(df[time_col]).dt.year
    return SplitMasks(
        train=_year_mask(years, train_years),
        validation=_year_mask(years, validation_years),
        test=_year_mask(years, test_years),
    )


def backtest_masks(
    df: pd.DataFrame, time_col: str, backtests: Sequence[dict]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Masks for expanding-window backtests, e.g. ``[{"train": [...], "evaluate": [...]}]``."""
    years = pd.to_datetime(df[time_col]).dt.year
    folds = []
    for fold in backtests:
        if max(fold["train"]) >= min(fold["evaluate"]):
            raise ValueError("each backtest must evaluate on years after its training years")
        folds.append((_year_mask(years, fold["train"]), _year_mask(years, fold["evaluate"])))
    return folds


def negative_subsample(y: Sequence[int], rate: float, seed: int = 42) -> np.ndarray:
    """Boolean mask that keeps all positives and a random share ``rate`` of negatives."""
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    y_arr = np.asarray(y)
    rng = np.random.default_rng(seed)
    keep_negative = rng.random(len(y_arr)) < rate
    return (y_arr == 1) | keep_negative
