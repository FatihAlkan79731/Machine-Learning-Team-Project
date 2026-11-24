"""Command-line downloads of the raw project data.

Examples
--------
python scripts/download_data.py stats19 --years 2018 2019 2020 2021 2022 2023 2024
python scripts/download_data.py weather --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import time

from uarps.config import load_config, project_path
from uarps.data import stats19, weather


def cmd_stats19(args: argparse.Namespace, cfg: dict) -> None:
    years = args.years or cfg["time"]["years"]
    dest = project_path(cfg["paths"]["raw"]) / "stats19"
    for path in stats19.download_collisions(years, dest, overwrite=args.overwrite):
        print(f"saved {path}")


def cmd_weather(args: argparse.Namespace, cfg: dict) -> None:
    dest = project_path(cfg["paths"]["raw"]) / "weather"
    dest.mkdir(parents=True, exist_ok=True)
    grid = weather.point_grid(cfg["study_area"]["bbox"], cfg["weather"]["grid_spacing_deg"])
    grid.to_csv(dest / "points.csv", index=False)
    for row in grid.itertuples(index=False):
        target = dest / f"point_{row.point_id:03d}.csv"
        if target.exists() and not args.overwrite:
            continue
        df = weather.fetch_hourly(
            row.latitude, row.longitude, args.start, args.end, cfg["weather"]["hourly_variables"]
        )
        df.to_csv(target, index=False)
        print(f"saved {target}")
        time.sleep(args.pause)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="source", required=True)

    p_stats = sub.add_parser("stats19", help="STATS19 collision files")
    p_stats.add_argument("--years", type=int, nargs="*")
    p_stats.add_argument("--overwrite", action="store_true")

    p_weather = sub.add_parser("weather", help="Open-Meteo hourly weather on a point grid")
    p_weather.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_weather.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_weather.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    p_weather.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    cfg = load_config(args.config)
    {"stats19": cmd_stats19, "weather": cmd_weather}[args.source](args, cfg)


if __name__ == "__main__":
    main()
