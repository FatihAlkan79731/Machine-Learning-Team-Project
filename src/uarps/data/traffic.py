"""DfT road traffic counts (AADF) and traffic exposure per cell.

Source: https://roadtraffic.dft.gov.uk/downloads ("Data at the count point level", AADF).
Download the AADF CSV manually and place it in ``data/raw/traffic/``.
The time-of-day profile comes from table TRA0307:
https://www.gov.uk/government/statistical-data-sets/road-traffic-estimates-tra
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AADF_REQUIRED_COLUMNS = [
    "count_point_id",
    "year",
    "region_name",
    "latitude",
    "longitude",
    "link_length_km",
    "all_motor_vehicles",
]

AADF_OPTIONAL_COLUMNS = [
    "road_name",
    "road_type",
    "estimation_method",
    "pedal_cycles",
    "two_wheeled_motor_vehicles",
    "all_hgvs",
]


def load_aadf(path: str | Path, region: str | None = "London") -> pd.DataFrame:
    """Load the AADF count-point file and optionally keep one region.

    Raises a clear error if the expected columns are missing, because the DfT file
    layout may change between releases.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in AADF_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"AADF file is missing columns {missing}; check the file layout")
    keep = AADF_REQUIRED_COLUMNS + [c for c in AADF_OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep]
    if region is not None:
        df = df[df["region_name"] == region]
    return df.reset_index(drop=True)


def daily_vehicle_km(aadf: pd.DataFrame, cell_col: str = "cell") -> pd.DataFrame:
    """Sum of AADF x link length per cell and year (vehicle-km per average day).

    ``aadf`` must already contain an H3 ``cell`` column for each count point.
    """
    df = aadf.copy()
    df["vehicle_km"] = df["all_motor_vehicles"] * df["link_length_km"]
    out = df.groupby([cell_col, "year"], as_index=False).agg(
        daily_vehicle_km=("vehicle_km", "sum"),
        n_count_points=("count_point_id", "nunique"),
    )
    if "all_hgvs" in df:
        hgv = df.assign(hgv_km=df["all_hgvs"] * df["link_length_km"])
        hgv = hgv.groupby([cell_col, "year"])["hgv_km"].sum().reset_index()
        out = out.merge(hgv, on=[cell_col, "year"], how="left")
        out["hgv_share"] = out["hgv_km"] / out["daily_vehicle_km"].where(
            out["daily_vehicle_km"] > 0
        )
        out = out.drop(columns="hgv_km")
    return out


def window_exposure(
    daily: pd.DataFrame, profile: pd.DataFrame, cell_col: str = "cell"
) -> pd.DataFrame:
    """Distribute daily vehicle-km over windows with a day-of-week x window profile (Eq. 7).

    ``profile`` has columns ``day_of_week`` (0 = Monday), ``window_of_day`` and ``share``,
    where ``share`` is the fraction of that day's traffic in the window (sums to 1 per day).
    """
    check = profile.groupby("day_of_week")["share"].sum()
    if not ((check - 1.0).abs() < 1e-6).all():
        raise ValueError("profile shares must sum to 1 for each day of the week")
    out = daily.merge(profile, how="cross")
    out["exposure_vkm"] = out["daily_vehicle_km"] * out["share"]
    return out[[cell_col, "year", "day_of_week", "window_of_day", "exposure_vkm"]]
