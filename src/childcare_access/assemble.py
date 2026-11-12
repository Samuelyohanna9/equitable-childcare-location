"""Common assembly of the model inputs.

The steps of the build that do not depend on where demand, vulnerability, facilities and candidates come
from: the walking network, snapping, demand locations, current access, candidate relations, the dominance screen,
feasibility and the saved inputs folder. They take tables (data frames), not source files, and are called by
``workflow.build`` for every configuration. ``checks`` is the build's check log
(``add(name, passed, detail)``, ``note(message)``); a failed check stops the build.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import geopandas as gpd
import networkx as nx
import pandas as pd

from .validation import sha256_file

if TYPE_CHECKING:  # imported lazily below: these modules read the active configuration
    from .all_demand_candidates import SparseCandidateRelations
    from .optimisation import DominanceScreen


class Checks(Protocol):
    """What the assembly steps need from the check recorder of a build (``workflow._Checks``)."""

    def add(self, name: str, passed: bool, detail: object) -> None:
        """Record one check (a failed check stops the build)."""

    def note(self, message: str) -> None:
        """Log a progress message."""


GRAPHS = ("walk_graph_base.pkl", "walk_graph_layered.pkl")


def walking_network(
    districts: gpd.GeoDataFrame, rule_set: str, snapshot: Path, out: Path, checks: Checks
) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
    """Pedestrian base graph from the OSM snapshot and its layered routing graph.

    The graph follows the rule set ``rule_set``; both graphs are saved with the network filter audit.
    """
    import networkx as nx

    from . import pedestrian_network as pn

    if rule_set == pn.VERSION_FINAL:
        graph, audit = pn.load_pedestrian_graph(districts, response_path=snapshot, way_rule=pn.way_class_sidewalk_proxy)
    elif rule_set == pn.VERSION_FULL:
        graph, audit = pn.load_pedestrian_graph(districts, response_path=snapshot)
    else:
        graph, audit = pn.load_pedestrian_graph(districts)
    graph.graph["pedestrian_network"] = rule_set
    layered, link_count = pn.layered_graph(graph)
    components = nx.number_strongly_connected_components(graph)
    checks.add(
        "pedestrian base graph strongly connected",
        components == 1,
        {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "rule_set": rule_set,
            "links_P_R": dict(link_count),
        },
    )
    for name, g in zip(GRAPHS, (graph, layered), strict=True):
        with (out / name).open("wb") as handle:
            pickle.dump(g, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "network_filter_audit.json").write_text(json.dumps(audit, indent=1, default=str), encoding="utf-8")
    return graph, layered


def on_facility_layer(frame: pd.DataFrame) -> pd.DataFrame:
    """Sites (facilities, candidates) attach to the facility-layer copy of their base network node."""
    from . import pedestrian_network as pn

    frame["base_node"] = frame["nearest_node"].astype("int64")
    frame["nearest_node"] = frame["base_node"].map(pn.facility_node)
    return frame


def snap(
    graph: nx.MultiDiGraph, buildings: pd.DataFrame | None, facilities: gpd.GeoDataFrame, checks: Checks
) -> tuple[pd.DataFrame | None, gpd.GeoDataFrame, pd.DataFrame | None]:
    """Snap buildings and existing facilities to the base graph.

    Buildings are snapped at ``rep_x``, ``rep_y``; ``buildings`` is None when the methods use none.
    """
    from .core import assign_nearest_nodes
    from .network import snap_buildings

    snapping = None
    if buildings is not None:
        snapping = snap_buildings(graph, buildings)
        buildings["nearest_node"] = snapping["nearest_node"].to_numpy()
        buildings["snap_distance_m"] = snapping["snap_distance_m"].to_numpy()
    facilities = on_facility_layer(assign_nearest_nodes(graph, facilities))
    if snapping is not None:
        checks.add(
            "building snapping",
            snapping["nearest_node"].notna().all(),
            {
                "median_m": round(float(snapping["snap_distance_m"].median()), 1),
                "max_m": round(float(snapping["snap_distance_m"].max()), 1),
            },
        )
    return buildings, facilities, snapping


def demand_locations(
    graph: nx.MultiDiGraph,
    origins: pd.DataFrame,
    districts: gpd.GeoDataFrame,
    groups: pd.DataFrame,
    totals: pd.Series,
    definition: str,
    checks: Checks,
    detail: dict,
) -> gpd.GeoDataFrame:
    """Demand locations (network node x district) from the standard demand-origin table.

    The table follows :mod:`schema`; origins without ``nearest_node`` are snapped here. Every district total
    ``totals`` must be conserved. Without an age split in the origins, the age columns of the demand locations stay
    empty (NaN).
    """
    import numpy as np

    from . import schema
    from . import validation as v
    from .core import DISTRICT_NAME
    from .demand import build_demand_nodes
    from .network import snap_buildings

    split = all(c in origins for c in ("age_0_2", "age_3_5"))
    schema.check(
        "demand_origins",
        origins[list(schema.COLUMNS["demand_origins"]) + (["age_0_2", "age_3_5"] if split else [])].rename(
            columns={DISTRICT_NAME: "district_name"}
        ),
        districts,
    )
    frame = origins.copy()
    if "nearest_node" not in frame:
        frame["nearest_node"] = np.nan
    todo = frame["nearest_node"].isna()
    if todo.any():
        points = pd.DataFrame(
            {
                "building_id": frame.loc[todo, "origin_id"],
                "rep_x": frame.loc[todo, "x"],
                "rep_y": frame.loc[todo, "y"],
            }
        )
        snapped = snap_buildings(graph, points)
        frame.loc[todo, "nearest_node"] = snapped["nearest_node"]
        frame.loc[todo, "snap_distance_m"] = snapped["snap_distance_m"]
    frame["nearest_node"] = frame["nearest_node"].astype("int64")
    sources = pd.DataFrame(
        {
            DISTRICT_NAME: frame[DISTRICT_NAME],
            "district_id": frame["district_id"],
            "footprint_m2": frame.get("footprint_m2", 0.0),
            "exclude_student_dormitory": frame.get("excluded", False),
        }
    )
    allocation = pd.DataFrame(
        {c: frame[c] if split else np.nan for c in ("age_0_2", "age_3_5")} | {"children_0_5": frame["children_0_5"]},
        index=frame.index,
    )
    demand = build_demand_nodes(graph, sources, allocation, frame[["nearest_node"]], districts, groups)
    if not split:
        demand[["age_0_2", "age_3_5"]] = np.nan
    demand["demand_definition"] = definition
    conservation = v.check_demand_conservation(demand, totals)
    checks.add(
        "demand locations conserve every district total",
        True,
        {
            "demand_locations": len(demand),
            "q4_locations": int(demand["least_advantaged"].sum()),
            **conservation,
            **detail,
        },
    )
    return demand


def relations_and_screen(
    layered: nx.MultiDiGraph,
    demand: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    p: int,
    out: Path,
    checks: Checks,
) -> tuple[pd.DataFrame, SparseCandidateRelations, DominanceScreen]:
    """Current access, sparse candidate relations, the Q4 dominance screen, model sizes and feasibility."""
    from . import validation as v
    from .network import build_current_access
    from .optimisation import (
        compute_sparse_candidate_relations,
        model_size_estimates,
        q4_dominance_screen,
    )

    located = int(demand["least_advantaged"].astype(bool).sum())
    if located == 0:
        raise RuntimeError(
            "no demand location lies in a priority-group district: the Rawlsian models would have no "
            "least-advantaged demand. Check the priority group and the demand coverage."
        )
    current = build_current_access(layered, demand, facilities)
    connectivity = v.check_network_connectivity(
        layered, demand["nearest_node"].astype("int64"), facilities["nearest_node"].astype("int64")
    )
    checks.add("every demand location reaches an existing facility", True, connectivity)
    checks.note("computing sparse candidate relations (the longest step) ...")
    relations = compute_sparse_candidate_relations(layered, demand, current, candidates)
    with (out / "sparse_relations_all_demand.pkl").open("wb") as handle:
        pickle.dump(relations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        screen = q4_dominance_screen(relations, demand, current, p_new=p)
    except RuntimeError as error:  # the screen's own message is terse; say what it means for this city
        raise RuntimeError(
            f"dominance screen: {error} The screen keeps, for the priority group, the candidates that are not "
            "dominated "
            f"and needs at least P = {p} of them. This city has {located} priority-group demand location(s) and "
            f"{len(candidates)} candidate(s). Reduce P, or check the priority group, its demand coverage and the "
            "candidate sites."
        ) from error
    checks.add(
        "Q4 dominance screen and dominator chains",
        screen.summary["mapping_chains_validated"] == screen.summary["strict_pointwise_dominated_candidates"],
        screen.summary,
    )
    screen.report.to_csv(out / "candidate_dominance_report.csv", index=False)
    screen.mapping.to_csv(out / "dominance_mapping.csv", index=False)
    (out / "dominance_summary.json").write_text(json.dumps(screen.summary, indent=2), encoding="utf-8")
    model_size_estimates(relations, demand, screen.retained_indices, current).to_csv(
        out / "model_size_estimates.csv", index=False
    )
    feasibility = v.check_optimisation_feasibility(
        p, candidates, current, relations, screened_count=len(screen.retained_indices)
    )
    checks.add("optimisation inputs feasible", True, feasibility)
    return current, relations, screen


def save(out: Path, bundle: dict, buildings: pd.DataFrame | None, children: pd.DataFrame, report: dict) -> None:
    """Write the inputs folder (the same format as the Münster model inputs)."""
    bundle["network"]["sha256"] = {name: sha256_file(out / name) for name in GRAPHS}
    with (out / "inputs_bundle.pkl").open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if buildings is not None:
        buildings.to_file(out / "buildings_alkis_allocated.gpkg", layer="buildings", driver="GPKG")
    demand = bundle["demand"]
    demand.drop(columns=[c for c in ["x", "y"] if c in demand]).to_file(
        out / "demand_nodes.gpkg", layer="demand", driver="GPKG"
    )
    bundle["current_access"].to_csv(out / "current_access.csv", index=False)
    children.to_csv(out / "children_by_district.csv", index=False)
    (out / "validation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
