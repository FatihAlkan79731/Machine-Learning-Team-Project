import pandas as pd
import pytest

h3 = pytest.importorskip("h3")

from uarps.features.spatial import (  # noqa: E402
    latlng_to_cells,
    parent_cells,
    ring_neighbours,
    spatial_lag,
)


def test_ring_sizes():
    cell = latlng_to_cells([51.5074], [-0.1278], 8)[0]
    assert len(ring_neighbours(cell, 1)) == 6
    assert len(ring_neighbours(cell, 2)) == 18


def test_parent_resolution():
    cell = latlng_to_cells([51.5074], [-0.1278], 8)[0]
    assert h3.get_resolution(parent_cells([cell], 6)[0]) == 6


def test_spatial_lag_mean_of_neighbours():
    centre = latlng_to_cells([51.5074], [-0.1278], 8)[0]
    neigh = sorted(ring_neighbours(centre, 1))
    values = pd.Series({centre: 100.0, **{n: float(i) for i, n in enumerate(neigh)}}, name="x")
    lag = spatial_lag(values, 1)
    assert lag[centre] == pytest.approx(sum(range(6)) / 6)
