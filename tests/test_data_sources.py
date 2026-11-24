import pandas as pd
import pytest

from uarps.data.stats19 import clean_collisions, filter_police_forces, load_collisions
from uarps.data.traffic import daily_vehicle_km, window_exposure
from uarps.data.weather import aggregate_to_windows, point_grid


def test_stats19_load_filter_clean(tmp_path):
    raw = pd.DataFrame(
        {
            "accident_index": ["A1", "A1", "A2", "A3", "A4"],
            "accident_year": [2019] * 5,
            "longitude": [-0.12, -0.12, -0.10, None, -2.0],
            "latitude": [51.50, 51.50, 51.52, 51.40, 53.4],
            "police_force": [1, 1, 48, 1, 7],
            "accident_severity": [3, 3, 2, 3, 3],
            "date": ["05/01/2019", "05/01/2019", "06/01/2019", "07/01/2019", "07/01/2019"],
            "time": ["08:15", "08:15", "23:59", "10:00", "11:00"],
        }
    )
    path = tmp_path / "collisions.csv"
    raw.to_csv(path, index=False)

    df = load_collisions([path])
    assert "collision_index" in df and "legacy_collision_severity" in df
    df = clean_collisions(filter_police_forces(df))
    assert df["collision_index"].tolist() == [
        "A1",
        "A2",
    ]  # duplicate, no-coords, non-London removed
    assert df["timestamp"].iloc[1] == pd.Timestamp("2019-01-06 23:59")


def test_weather_grid_and_window_aggregation():
    grid = point_grid([-0.5, 51.3, 0.3, 51.7], 0.1)
    assert len(grid) == 9 * 5
    hourly = pd.DataFrame(
        {
            "time": pd.date_range("2023-01-01", periods=6, freq="h"),
            "precipitation": [0.0, 1.0, 2.0, 0.5, 0.0, 0.0],
            "temperature_2m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "wind_speed_10m": [5, 9, 7, 1, 2, 3],
        }
    )
    out = aggregate_to_windows(hourly, 3)
    assert out["precipitation"].tolist() == [3.0, 0.5]
    assert out["temperature_2m"].tolist() == [2.0, 5.0]
    assert out["wind_speed_10m"].tolist() == [9, 3]
    assert out["precipitation_prev_window"].iloc[1] == 3.0


def test_traffic_exposure_sums_to_daily_total():
    aadf = pd.DataFrame(
        {
            "cell": ["c1", "c1", "c2"],
            "year": [2023] * 3,
            "count_point_id": [1, 2, 3],
            "all_motor_vehicles": [10_000, 5_000, 20_000],
            "link_length_km": [0.5, 1.0, 0.25],
            "all_hgvs": [1_000, 0, 2_000],
        }
    )
    daily = daily_vehicle_km(aadf)
    c1 = daily.set_index("cell").loc["c1"]
    assert c1["daily_vehicle_km"] == pytest.approx(10_000)
    assert c1["hgv_share"] == pytest.approx(0.05)
    profile = pd.DataFrame(
        [(d, w, 1 / 8) for d in range(7) for w in range(8)],
        columns=["day_of_week", "window_of_day", "share"],
    )
    exp = window_exposure(daily, profile)
    total = exp[(exp["cell"] == "c1") & (exp["day_of_week"] == 0)]["exposure_vkm"].sum()
    assert total == pytest.approx(10_000)


def test_traffic_profile_must_sum_to_one():
    daily = pd.DataFrame({"cell": ["c"], "year": [2023], "daily_vehicle_km": [1.0]})
    bad = pd.DataFrame({"day_of_week": [0], "window_of_day": [0], "share": [0.5]})
    with pytest.raises(ValueError):
        window_exposure(daily, bad)
