"""Road network and point-of-interest features from OpenStreetMap (via OSMnx).

Data (c) OpenStreetMap contributors, Open Database Licence (ODbL).
Requires ``osmnx``, ``geopandas`` and ``h3``. Downloading the full Greater London
network takes several minutes; cache the result in ``data/external/``.
"""

from __future__ import annotations

import pandas as pd

ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]

POI_TAGS = {
    "amenity": ["school", "pub", "bar", "nightclub"],
    "shop": True,
    "highway": ["traffic_signals", "crossing", "bus_stop"],
    "junction": ["roundabout"],
}


def download_drive_network(place: str = "Greater London, UK"):
    """Download the drivable street network as an OSMnx graph."""
    import osmnx as ox  # noqa: PLC0415

    return ox.graph_from_place(place, network_type="drive", simplify=True)


def download_pois(place: str = "Greater London, UK"):
    """Download points of interest and road furniture listed in ``POI_TAGS``."""
    import osmnx as ox  # noqa: PLC0415

    return ox.features_from_place(place, tags=POI_TAGS)


def edge_length_by_class(edges: pd.DataFrame, cell_col: str = "cell") -> pd.DataFrame:
    """Total road length (km) per cell and OSM highway class.

    ``edges`` must contain ``highway`` (str or list), ``length`` in metres and an H3
    ``cell`` column (for example the cell of each edge's midpoint).
    """
    df = edges[[cell_col, "highway", "length"]].copy()
    df["highway"] = df["highway"].apply(lambda h: h[0] if isinstance(h, list) else h)
    df["highway"] = df["highway"].str.replace("_link", "", regex=False)
    df = df[df["highway"].isin(ROAD_CLASSES)]
    out = df.pivot_table(
        index=cell_col, columns="highway", values="length", aggfunc="sum", fill_value=0.0
    )
    out = out.reindex(columns=ROAD_CLASSES, fill_value=0.0) / 1000.0
    out.columns = [f"road_km_{c}" for c in out.columns]
    return out.reset_index()
