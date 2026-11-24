"""STATS19 collision data: download, loading, filtering and cleaning.

Source: Department for Transport, Road Safety Data (Open Government Licence v3.0)
https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import requests

URL_TEMPLATE = (
    "https://data.dft.gov.uk/road-accidents-safety-data/"
    "dft-road-casualty-statistics-collision-{year}.csv"
)

# STATS19 police_force codes for Greater London
LONDON_POLICE_FORCES: tuple[int, ...] = (1, 48)

# Older files use "accident_*" names; newer files use "collision_*".
COLUMN_ALIASES = {
    "accident_index": "collision_index",
    "accident_year": "collision_year",
    "accident_reference": "collision_reference",
    "accident_severity": "legacy_collision_severity",
    "collision_severity": "legacy_collision_severity",
}

# Columns kept for the project. Scene attributes (light, weather, road surface) are kept
# for EDA only and must never be used as model inputs (see gap G1 in the paper).
KEEP_COLUMNS = [
    "collision_index",
    "collision_year",
    "longitude",
    "latitude",
    "police_force",
    "legacy_collision_severity",
    "number_of_vehicles",
    "number_of_casualties",
    "date",
    "time",
    "road_type",
    "speed_limit",
    "junction_detail",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
    "lsoa_of_collision_location",
]

EDA_ONLY_COLUMNS = ["light_conditions", "weather_conditions", "road_surface_conditions"]


def download_collisions(
    years: Iterable[int], dest_dir: str | Path, overwrite: bool = False, timeout: int = 120
) -> list[Path]:
    """Download yearly STATS19 collision CSV files into ``dest_dir``."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in years:
        url = URL_TEMPLATE.format(year=year)
        target = dest / Path(url).name
        if target.exists() and not overwrite:
            paths.append(target)
            continue
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            tmp = target.with_suffix(".part")
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
            tmp.rename(target)
        paths.append(target)
    return paths


def load_collisions(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Read and concatenate yearly collision files, harmonising column names."""
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
        frames.append(df[[c for c in KEEP_COLUMNS if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def filter_police_forces(
    df: pd.DataFrame, forces: Iterable[int] = LONDON_POLICE_FORCES
) -> pd.DataFrame:
    """Keep collisions recorded by the given police forces."""
    return df[pd.to_numeric(df["police_force"], errors="coerce").isin(list(forces))].copy()


def clean_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps, drop records without valid location or time, remove duplicates.

    Adds a ``timestamp`` column. ``date`` is ``dd/mm/yyyy`` and ``time`` is ``HH:MM``.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(
        out["date"].astype(str) + " " + out["time"].astype(str),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    valid = out["timestamp"].notna() & out["latitude"].between(49.0, 61.0)
    valid &= out["longitude"].between(-9.0, 2.5)
    out = out[valid]
    return out.drop_duplicates(subset="collision_index").reset_index(drop=True)
