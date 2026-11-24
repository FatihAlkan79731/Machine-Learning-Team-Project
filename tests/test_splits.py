import numpy as np
import pandas as pd
import pytest

from uarps.evaluation.splits import backtest_masks, negative_subsample, temporal_split


@pytest.fixture
def frame():
    times = pd.date_range("2018-01-01", "2024-12-31 21:00", freq="7D")
    return pd.DataFrame({"window_start": times})


def test_temporal_split_is_disjoint_and_ordered(frame):
    m = temporal_split(frame, "window_start", [2018, 2019, 2020, 2021, 2022], [2023], [2024])
    assert not (m.train & m.validation).any()
    assert not (m.validation & m.test).any()
    years = frame["window_start"].dt.year
    assert years[m.train].max() < years[m.validation].min() < years[m.test].min()


def test_temporal_split_rejects_leakage(frame):
    with pytest.raises(ValueError):
        temporal_split(frame, "window_start", [2018, 2023], [2022], [2024])


def test_backtests(frame):
    folds = backtest_masks(
        frame, "window_start", [{"train": [2018, 2019, 2020], "evaluate": [2021]}]
    )
    train, ev = folds[0]
    assert frame["window_start"].dt.year[ev].unique().tolist() == [2021]
    assert not (train & ev).any()


def test_negative_subsample_keeps_all_positives():
    rng = np.random.default_rng(5)
    y = (rng.random(100_000) < 0.01).astype(int)
    keep = negative_subsample(y, 0.05, seed=1)
    assert keep[y == 1].all()
    assert keep[y == 0].mean() == pytest.approx(0.05, abs=0.005)
