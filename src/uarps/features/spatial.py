"""H3 grid helpers and spatial-lag features.

Requires the ``h3`` package (version 4 API).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _h3():
    try:
        import h3  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Install the h3 package: pip install 'h3>=4'") from exc
    return h3


def latlng_to_cells(lat: Iterable[float], lng: Iterable[float], resolution: int = 8) -> list[str]:
    """Map coordinates to H3 cell identifiers at the given resolution."""
    h3 = _h3()
    return [h3.latlng_to_cell(a, b, resolution) for a, b in zip(lat, lng, strict=True)]


def ring_neighbours(cell: str, k: int) -> set[str]:
    """All cells within ``k`` hexagonal steps of ``cell``, excluding the cell itself."""
    h3 = _h3()
    return set(h3.grid_disk(cell, k)) - {cell}


def parent_cells(cells: Iterable[str], resolution: int) -> list[str]:
    """Parent cell of each cell at a coarser resolution (used for spatial block checks)."""
    h3 = _h3()
    return [h3.cell_to_parent(c, resolution) for c in cells]


def spatial_lag(values: pd.Series, k: int = 1) -> pd.Series:
    """Mean of a cell-level value over the k-ring neighbours (Eq. 6 in the paper).

    ``values`` is indexed by H3 cell. Neighbours that are not in the index (for
    example cells outside the study area) are ignored.
    """
    lookup = values.to_dict()
    result = {}
    for cell in values.index:
        neigh = [lookup[n] for n in ring_neighbours(cell, k) if n in lookup]
        result[cell] = float(np.mean(neigh)) if neigh else np.nan
    return pd.Series(result, name=f"{values.name}_lag{k}" if values.name else f"lag{k}")
