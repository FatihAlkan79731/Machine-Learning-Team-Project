import numpy as np
import pandas as pd
import pytest

from uarps.features.temporal import add_time_features, cyclical_encode, window_of_day, window_start


def test_window_of_day_three_hour_bins():
    ts = pd.Series(
        pd.to_datetime(
            ["2023-01-02 00:00", "2023-01-02 02:59", "2023-01-02 03:00", "2023-01-02 23:30"]
        )
    )
    assert window_of_day(ts, 3).tolist() == [0, 0, 1, 7]


def test_window_of_day_rejects_invalid_width():
    with pytest.raises(ValueError):
        window_of_day(pd.Series(pd.to_datetime(["2023-01-01"])), 5)


def test_window_start_floors_to_window():
    ts = pd.Series(pd.to_datetime(["2023-05-01 14:47"]))
    assert window_start(ts, 3).iloc[0] == pd.Timestamp("2023-05-01 12:00")


def test_cyclical_encoding_wraps_around():
    s, c = cyclical_encode([0, 7], 8)
    # last window is close to the first one on the unit circle
    dist = np.hypot(s[0] - s[1], c[0] - c[1])
    assert dist == pytest.approx(2 * np.sin(np.pi / 8))
    assert s[0] == pytest.approx(0.0) and c[0] == pytest.approx(1.0)


def test_add_time_features_flags():
    df = pd.DataFrame(
        {
            "window_start": pd.to_datetime(
                ["2020-04-04 09:00", "2020-12-25 12:00", "2022-06-01 21:00"]
            )
        }
    )
    out = add_time_features(
        df,
        holidays_dates=[pd.Timestamp("2020-12-25")],
        covid_periods=[("2020-03-23", "2020-07-04")],
    )
    assert out["is_weekend"].tolist() == [1, 0, 0]  # 4 Apr 2020 was a Saturday
    assert out["is_bank_holiday"].tolist() == [0, 1, 0]
    assert out["is_covid_restriction"].tolist() == [1, 0, 0]
    assert out.loc[0, "window_of_week"] == 5 * 8 + 3
