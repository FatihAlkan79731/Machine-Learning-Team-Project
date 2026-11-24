"""Time-window assignment and calendar features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def window_of_day(timestamps: pd.Series, window_hours: int = 3) -> pd.Series:
    """Index of the time window within the day (0 .. 24 / window_hours - 1)."""
    if 24 % window_hours != 0:
        raise ValueError("window_hours must divide 24")
    return (pd.to_datetime(timestamps).dt.hour // window_hours).astype("int16")


def window_start(timestamps: pd.Series, window_hours: int = 3) -> pd.Series:
    """Start time of the window that contains each timestamp."""
    return pd.to_datetime(timestamps).dt.floor(f"{window_hours}h")


def cyclical_encode(
    values: Iterable[float] | pd.Series, period: float
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a periodic variable as sine and cosine (Eq. 4 in the paper).

    Parameters
    ----------
    values:
        Position within the cycle, from 0 to ``period - 1``.
    period:
        Length of the cycle, for example 8 windows per day or 12 months.
    """
    arr = np.asarray(values, dtype=float)
    angle = 2.0 * np.pi * arr / period
    return np.sin(angle), np.cos(angle)


def add_time_features(
    df: pd.DataFrame,
    time_col: str = "window_start",
    window_hours: int = 3,
    holidays_dates: Iterable[pd.Timestamp] | None = None,
    covid_periods: Iterable[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Add calendar features for each row's window start time.

    Adds window of day, day of week, month, weekend flag, cyclical encodings and,
    if given, bank-holiday and COVID-19 restriction flags.
    """
    out = df.copy()
    ts = pd.to_datetime(out[time_col])
    windows_per_day = 24 // window_hours

    out["window_of_day"] = window_of_day(ts, window_hours)
    out["day_of_week"] = ts.dt.dayofweek.astype("int16")
    out["month"] = ts.dt.month.astype("int16")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["window_of_week"] = (out["day_of_week"] * windows_per_day + out["window_of_day"]).astype(
        "int16"
    )

    out["wod_sin"], out["wod_cos"] = cyclical_encode(out["window_of_day"], windows_per_day)
    out["month_sin"], out["month_cos"] = cyclical_encode(out["month"] - 1, 12)

    if holidays_dates is not None:
        holiday_set = {pd.Timestamp(d).normalize() for d in holidays_dates}
        out["is_bank_holiday"] = ts.dt.normalize().isin(holiday_set).astype("int8")

    if covid_periods is not None:
        flag = np.zeros(len(out), dtype="int8")
        for start, end in covid_periods:
            in_period = (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))
            flag |= in_period.to_numpy().astype("int8")
        out["is_covid_restriction"] = flag

    return out
