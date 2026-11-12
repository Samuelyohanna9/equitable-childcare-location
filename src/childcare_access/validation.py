"""Reusable input and solution checks.

Each check raises :class:`ValidationError` with a message that names the problem, how often it occurs and a few
examples, and otherwise returns a small report (dict). None of them depends on a particular city: expected values
(district totals, CRS, budget) are passed in.

- schemas and CRS: :func:`require_columns`, :func:`require_crs`
- demand: :func:`check_demand_conservation`
- spatial joins: :func:`check_district_membership`
- network: :func:`check_network_connectivity`
- candidates: :func:`check_candidate_sites`
- optimisation: :func:`check_optimisation_feasibility`, :func:`check_solution`
- checksums: :func:`sha256_file`
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .all_demand_candidates import SparseCandidateRelations


class ValidationError(ValueError):
    """An input or solution violates a requirement of the method."""


def _examples(values: Iterable[object], limit: int = 5) -> str:
    """The first few values, for an error message."""
    values = list(values)
    shown = ", ".join(map(str, values[:limit]))
    return shown + (f", ... ({len(values)} in total)" if len(values) > limit else "")


# schemas and CRS
def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Stop with a clear message if a table lacks required columns."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"{name}: missing column(s) {missing}; available: {sorted(map(str, frame.columns))[:20]}")


def require_crs(
    frame: gpd.GeoDataFrame | gpd.GeoSeries,
    name: str,
    expected: int | None = None,
    projected: bool = True,
) -> None:
    """The layer has a CRS, it is the expected one (EPSG code) and, if ``projected``, uses metres."""
    if frame.crs is None:
        raise ValidationError(f"{name}: no CRS defined")
    if expected is not None and frame.crs.to_epsg() != expected:
        raise ValidationError(f"{name}: CRS is EPSG:{frame.crs.to_epsg()}, expected EPSG:{expected}")
    if projected:
        units = {axis.unit_name for axis in frame.crs.axis_info}
        if not frame.crs.is_projected or not units <= {"metre"}:
            raise ValidationError(f"{name}: CRS {frame.crs.to_string()} is not a projected CRS in metres")


# demand
def check_demand_conservation(
    demand: pd.DataFrame,
    district_totals: pd.Series,
    demand_column: str = "estimated_demand",
    district_column: str = "district_id",
    tolerance: float = 1e-6,
) -> dict:
    """Allocated demand sums to the district total in every district; no negative or missing demand."""
    require_columns(demand, [demand_column, district_column], "demand")
    values = demand[demand_column]
    if values.isna().any() or (values < 0).any():
        bad = demand.index[values.isna() | (values < 0)]
        raise ValidationError(
            f"demand: {len(bad)} missing or negative values in {demand_column!r} (e.g. {_examples(bad)})"
        )
    totals = district_totals.astype(float)
    allocated = demand.groupby(district_column)[demand_column].sum().reindex(totals.index, fill_value=0.0)
    extra = sorted(set(demand[district_column]) - set(totals.index))
    if extra:
        raise ValidationError(f"demand: {len(extra)} district ID(s) without a district total: {_examples(extra)}")
    error = (allocated - totals).abs()
    if error.max() > tolerance:
        worst = error.sort_values(ascending=False)
        raise ValidationError(
            f"demand: district totals not conserved in {int((error > tolerance).sum())} district(s); "
            f"largest error {worst.iloc[0]:.6g} in district {worst.index[0]} "
            f"(allocated {allocated[worst.index[0]]:.6g}, total {totals[worst.index[0]]:.6g})"
        )
    return {
        "districts": len(totals),
        "total": float(allocated.sum()),
        "max_abs_error": float(error.max()),
    }


# spatial joins
def check_district_membership(
    points: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, district_column: str = "district_id"
) -> dict:
    """Every point lies in exactly one district polygon, and that district is the one recorded for it."""
    require_columns(points, [district_column], "points")
    require_columns(districts, [district_column], "districts")
    if points.crs != districts.crs:
        districts = districts.to_crs(points.crs)
    joined = gpd.sjoin(
        points[[district_column, "geometry"]],
        districts[[district_column, "geometry"]],
        predicate="within",
        how="left",
        lsuffix="recorded",
        rsuffix="spatial",
    )
    hits = joined.groupby(level=0).size()
    outside = joined.index[joined[f"{district_column}_spatial"].isna()].unique()
    multiple = hits.index[hits > 1]
    if len(outside) or len(multiple):
        raise ValidationError(
            f"district membership: {len(outside)} point(s) outside every district (e.g. {_examples(outside)}), "
            f"{len(multiple)} in more than one (e.g. {_examples(multiple)})"
        )
    wrong = joined.index[
        joined[f"{district_column}_recorded"].astype(str) != joined[f"{district_column}_spatial"].astype(str)
    ]
    if len(wrong):
        raise ValidationError(
            f"district membership: {len(wrong)} point(s) recorded in another district than the one "
            f"containing them (e.g. {_examples(wrong)})"
        )
    return {"points": len(points), "districts": int(points[district_column].nunique())}


# network
def check_network_connectivity(
    graph: nx.DiGraph,
    demand_nodes: Iterable[int],
    facility_nodes: Iterable[int],
    weight: str = "walk_time_min",
    base_graph: nx.DiGraph | None = None,
) -> dict:
    """Check that the network can serve every demand location.

    All nodes exist, edge weights are finite and non-negative, every demand node reaches an existing facility, and
    (if given) the base pedestrian graph is strongly connected.
    """
    demand_nodes, facility_nodes = list(demand_nodes), list(facility_nodes)
    absent = [n for n in set(demand_nodes) | set(facility_nodes) if n not in graph]
    if absent:
        raise ValidationError(
            f"network: {len(absent)} demand/facility node(s) not in the routing graph (e.g. {_examples(absent)})"
        )
    bad = [(u, v) for u, v, w in graph.edges(data=weight) if w is None or not np.isfinite(w) or w < 0]
    if bad:
        raise ValidationError(
            f"network: {len(bad)} edge(s) with a missing, infinite or negative {weight!r} (e.g. {_examples(bad)})"
        )
    reverse = graph.reverse(copy=False) if graph.is_directed() else graph
    reached = nx.multi_source_dijkstra_path_length(reverse, set(facility_nodes), weight=weight)
    unreachable = sorted({n for n in demand_nodes if n not in reached})
    if unreachable:
        raise ValidationError(
            f"network: {len(unreachable)} demand node(s) cannot reach any existing facility (e.g. "
            f"{_examples(unreachable)})"
        )
    report = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "demand_nodes_reaching_a_facility": len(set(demand_nodes)),
    }
    if base_graph is not None:
        components = (
            nx.number_strongly_connected_components(base_graph)
            if base_graph.is_directed()
            else nx.number_connected_components(base_graph)
        )
        if components != 1:
            raise ValidationError(
                f"network: the base pedestrian graph has {components} (strongly) connected components, expected 1"
            )
        report["base_graph_components"] = 1
    return report


# candidates
def check_candidate_sites(
    candidates: pd.DataFrame,
    graph: nx.DiGraph,
    id_column: str = "facility_id",
    node_column: str = "nearest_node",
) -> dict:
    """Candidate IDs are unique and present, and every candidate is attached to a node of the routing graph."""
    require_columns(candidates, [id_column, node_column], "candidates")
    ids = candidates[id_column]
    if ids.isna().any() or ids.duplicated().any():
        raise ValidationError(
            f"candidates: {int(ids.isna().sum())} missing and {int(ids.duplicated().sum())} duplicated ID(s) "
            f"(e.g. {_examples(ids[ids.duplicated()])})"
        )
    absent = candidates.loc[~candidates[node_column].isin(list(graph.nodes)), id_column]
    if len(absent):
        raise ValidationError(
            f"candidates: {len(absent)} site(s) attached to nodes not in the routing graph (e.g. {_examples(absent)})"
        )
    return {"candidates": len(candidates)}


# optimisation
def check_optimisation_feasibility(
    p: int,
    candidates: pd.DataFrame,
    current_access: pd.DataFrame,
    relations: SparseCandidateRelations = None,
    q4_column: str = "least_advantaged",
    screened_count: int | None = None,
    id_column: str = "facility_id",
) -> dict:
    """The facility budget is achievable and the model data are consistent.

    - 1 <= P <= number of candidates (and <= screened candidates for the district model, if given);
    - every demand location has a finite, non-negative current walking time;
    - the least-advantaged group is not empty and has positive demand in each of its districts;
    - the sparse relations reference known candidates and demand locations, and every pair is a strict
      improvement on the current walking time (as the formulations assume).
    """
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValidationError(f"optimisation: facility budget P must be an integer >= 1, got {p!r}")
    if p > len(candidates):
        raise ValidationError(f"optimisation: P = {p} exceeds the {len(candidates)} candidate sites")
    if screened_count is not None and p > screened_count:
        raise ValidationError(f"optimisation: P = {p} exceeds the {screened_count} candidates kept by the screen")
    require_columns(
        current_access,
        ["demand_id", "current_walk_time", "estimated_demand", q4_column],
        "current access",
    )
    times = current_access["current_walk_time"]
    if not np.isfinite(times).all() or (times < 0).any():
        bad = current_access.loc[~np.isfinite(times) | (times < 0), "demand_id"]
        raise ValidationError(
            f"optimisation: {len(bad)} demand location(s) without a finite current walking time (e.g. {_examples(bad)})"
        )
    q4 = current_access[current_access[q4_column].astype(bool)]
    if q4.empty:
        raise ValidationError("optimisation: the least-advantaged group has no demand locations")
    district = "district_name" if "district_name" in q4 else None
    if district:
        demand = q4.groupby(district)["estimated_demand"].sum()
        if (demand <= 0).any():
            raise ValidationError(
                f"optimisation: least-advantaged district(s) without demand: {_examples(demand.index[demand <= 0])}"
            )
    report = {
        "P": int(p),
        "candidates": len(candidates),
        "demand_locations": len(current_access),
        "q4_locations": len(q4),
    }
    if relations is not None:
        known = set(candidates[id_column])
        unknown = [c for c in relations.candidate_ids if c not in known]
        if unknown:
            raise ValidationError(
                f"optimisation: {len(unknown)} relation candidate(s) not in the candidate table (e.g. "
                f"{_examples(unknown)})"
            )
        current = current_access.set_index("demand_id")["current_walk_time"]
        foreign = [i for i in relations.pairs_by_demand if i not in current.index]
        if foreign:
            raise ValidationError(
                f"optimisation: relations for {len(foreign)} unknown demand location(s) (e.g. {_examples(foreign)})"
            )
        n = len(relations.candidate_ids)
        bad = [
            (i, j)
            for i, pairs in relations.pairs_by_demand.items()
            for j, t in pairs
            if not 0 <= j < n or not 0 <= t < current[i]
        ]
        if bad:
            raise ValidationError(
                f"optimisation: {len(bad)} relation pair(s) out of range or not improving on the current "
                f"walking time (e.g. {_examples(bad)})"
            )
        report["relation_pairs"] = sum(len(v) for v in relations.pairs_by_demand.values())
    return report


def check_solution(sites: Iterable[str], p: int, candidates: pd.DataFrame, id_column: str = "facility_id") -> dict:
    """A solution opens exactly P distinct, known candidate sites."""
    sites = list(sites)
    unknown = [s for s in sites if s not in set(candidates[id_column])]
    if len(sites) != p or len(set(sites)) != len(sites) or unknown:
        raise ValidationError(
            f"solution: expected {p} distinct known sites, got {len(sites)} "
            f"({len(set(sites))} distinct, unknown: {unknown})"
        )
    return {"sites": sorted(sites)}


# checksums
def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
