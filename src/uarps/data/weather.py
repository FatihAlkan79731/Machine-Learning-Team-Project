"""Hourly historical weather from the Open-Meteo Historical Weather API.

Source: https://open-meteo.com/en/docs/historical-weather-api (data under CC BY 4.0).
No API key is required. Please keep request volume moderate: query a coarse grid of
points over the study area (about 0.1 degree spacing) and assign each H3 cell to the
nearest point.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
]

# How each variable is aggregated from hourly values to a multi-hour window
AGGREGATION = {
    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
    "precipitation": "sum",
    "rain": "sum",
    "snowfall": "sum",
    "cloud_cover": "mean",
    "wind_speed_10m": "max",
    "wind_gusts_10m": "max",
}


def point_grid(bbox: Sequence[float], spacing_deg: float = 0.1) -> pd.DataFrame:
    """Regular grid of query points covering ``bbox = (min_lon, min_lat, max_lon, max_lat)``."""
    min_lon, min_lat, max_lon, max_lat = bbox

    def axis(lo: float, hi: float) -> np.ndarray:
        n = int(np.ceil((hi - lo) / spacing_deg - 1e-9)) + 1
        return lo + spacing_deg * np.arange(n)

    lons = axis(min_lon, max_lon)
    lats = axis(min_lat, max_lat)
    grid = [(round(la, 4), round(lo, 4)) for la in lats for lo in lons]
    df = pd.DataFrame(grid, columns=["latitude", "longitude"])
    df.insert(0, "point_id", range(len(df)))
    return df


def fetch_hourly(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: Sequence[str] = DEFAULT_HOURLY,
    timezone: str = "Europe/London",
    timeout: int = 120,
) -> pd.DataFrame:
    """Download hourly weather for one point between two dates (inclusive, ``YYYY-MM-DD``)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": timezone,
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def aggregate_to_windows(hourly: pd.DataFrame, window_hours: int = 3) -> pd.DataFrame:
    """Aggregate hourly weather to windows using the rules in ``AGGREGATION``.

    Also adds ``precipitation_prev_window`` (precipitation in the preceding window), used
    as a proxy for wet road surfaces.
    """
    agg = {c: AGGREGATION.get(c, "mean") for c in hourly.columns if c != "time"}
    out = (
        hourly.set_index("time")
        .resample(f"{window_hours}h", label="left", closed="left")
        .agg(agg)
        .reset_index()
        .rename(columns={"time": "window_start"})
    )
    if "precipitation" in out:
        out["precipitation_prev_window"] = out["precipitation"].shift(1)
    return out
