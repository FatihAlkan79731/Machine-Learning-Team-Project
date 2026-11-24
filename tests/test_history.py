import math

import numpy as np
import pandas as pd
import pytest

from uarps.features.history import (
    add_history_features,
    decay_constant_from_half_life,
    decayed_intensity,
    past_window_sum,
)


def brute_force(counts, kappa):
    n = len(counts)
    return np.array(
        [sum(math.exp(-(t - tau) / kappa) * counts[tau] for tau in range(t)) for t in range(n)]
    )


def test_decayed_intensity_matches_definition():
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.3, size=200)
    kappa = 12.5
    np.testing.assert_allclose(decayed_intensity(counts, kappa), brute_force(counts, kappa))


def test_decayed_intensity_excludes_current_window():
    counts = np.array([0, 0, 5, 0])
    h = decayed_intensity(counts, 10.0)
    assert h[2] == 0.0  # the collision in window 2 is not visible in window 2
    assert h[3] > 0.0


def test_half_life():
    kappa = decay_constant_from_half_life(8)
    assert math.exp(-8 / kappa) == pytest.approx(0.5)


def test_past_window_sum_uses_only_past():
    counts = np.array([1, 0, 2, 3, 0])
    np.testing.assert_array_equal(past_window_sum(counts, 2), [0, 1, 1, 2, 5])


def test_add_history_features_per_cell():
    times = pd.date_range("2023-01-01", periods=16, freq="3h")
    panel = pd.DataFrame(
        {
            "cell": ["a"] * 16 + ["b"] * 16,
            "window_start": list(times) * 2,
            "n_collisions": [1] + [0] * 15 + [0] * 16,
        }
    )
    out = add_history_features(panel, history_days=(1,), half_life_days=(1,))
    a = out[out["cell"] == "a"]
    b = out[out["cell"] == "b"]
    assert a["hist_count_1d"].iloc[0] == 0  # nothing before the first window
    assert a["hist_count_1d"].iloc[1:9].eq(1).all()  # visible for the next 8 windows (1 day)
    assert a["hist_count_1d"].iloc[9] == 0
    assert b["hist_count_1d"].eq(0).all()  # no leakage across cells
