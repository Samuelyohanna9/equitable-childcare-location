"""Pedestrian network data access, building snapping and current accessibility.

The pedestrian-access rules and the layered routing graph are in :mod:`childcare_access.pedestrian_network`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .core import (
    DISTRICT_NAME,
    PROJECTED_CRS,
    WGS84,
    assign_nearest_nodes,
    compute_current_accessibility,
    compute_travel_time_matrix,
)


@contextlib.contextmanager
def osm_cache_only() -> Iterator[None]:
    """Read OSMnx data from the project cache with all HTTP requests blocked."""
    import osmnx as ox
    import requests

    def _blocked(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError(
            "Network access is blocked for the primary_2025_26 build: the OSMnx "
            "cache did not contain the requested query."
        )

    saved = (requests.post, requests.get, requests.Session.request, ox.settings.use_cache)
    requests.post = requests.get = requests.Session.request = _blocked
    ox.settings.use_cache = True
    try:
        yield
    finally:
        requests.post, requests.get, requests.Session.request, ox.settings.use_cache = saved


def snap_buildings(graph: nx.MultiDiGraph, buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """Nearest walking-network node for every building representative point, with distance."""
    points = gpd.GeoDataFrame(
        {"building_id": buildings["building_id"].to_numpy()},
        geometry=gpd.points_from_xy(buildings["rep_x"], buildings["rep_y"]),
        crs=PROJECTED_CRS,
    )
    snapped = assign_nearest_nodes(graph, points)
    node_xy = {node: (data["x"], data["y"]) for node, data in graph.nodes(data=True)}
    node_points = gpd.GeoSeries(
        gpd.points_from_xy(
            [node_xy[node][0] for node in snapped["nearest_node"]],
            [node_xy[node][1] for node in snapped["nearest_node"]],
        ),
        crs=graph.graph.get("crs", WGS84),
    ).to_crs(PROJECTED_CRS)
    distance = node_points.distance(points.geometry.reset_index(drop=True)).to_numpy()
    return pd.DataFrame(
        {
            "building_id": points["building_id"].to_numpy(),
            "nearest_node": snapped["nearest_node"].to_numpy(),
            "snap_distance_m": distance,
        },
        index=buildings.index,
    )


def build_current_access(graph: nx.MultiDiGraph, demand: gpd.GeoDataFrame, kitas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Walking time from every demand location to its nearest existing Kita."""
    matrix = compute_travel_time_matrix(graph, demand, kitas)
    access = compute_current_accessibility(demand, kitas, DISTRICT_NAME, matrix)
    if not np.isfinite(access["current_walk_time"]).all():
        raise ValueError("Some demand locations cannot reach any existing Kita.")
    return access


def load_kitas_from_file(path: str | Path) -> gpd.GeoDataFrame:
    """Existing facilities from a local vector file (points or polygons).

    In the format of ``download_kindergartens``: WGS84 centroid points with ``site_name``, ``facility_type`` and
    ``facility_id``.
    """
    raw = gpd.read_file(path)
    if raw.crs is None:
        raise ValueError(f"{path}: no CRS defined")
    points = raw.to_crs(PROJECTED_CRS).geometry.centroid.to_crs(WGS84)
    result = gpd.GeoDataFrame({"geometry": points.to_numpy()}, geometry="geometry", crs=WGS84)
    names = raw["name"] if "name" in raw.columns else pd.Series([None] * len(raw))
    result["site_name"] = [n if isinstance(n, str) and n else f"Existing Kita {i + 1}" for i, n in enumerate(names)]
    result["facility_type"] = "existing"
    result["facility_id"] = [f"existing_{i}" for i in range(len(result))]
    return result
