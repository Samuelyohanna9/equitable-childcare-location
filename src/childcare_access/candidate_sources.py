"""Candidate sources: where new facilities may be placed.

- ``residential_buildings`` (default): one candidate per network node of an eligible residential building
  (``optimisation.build_candidate_pool`` on the footprint-allocated building nodes).
- ``file``: supplied sites (points, or polygons reduced by the configured ``polygon_point`` rule; a centroid is used
  only when configured and only if it lies inside its polygon). A site is a candidate only if the source says it is
  eligible (``eligible_column``/``eligible_values``; without them every supplied site counts as eligible AS SUPPLIED). A
  site exactly on a district boundary stops the build (``boundary_site = "error"``, default) or takes the lowest
  district ID (``"lowest_id"``, recorded per site). Sites outside every district, too far from the network
  (``max_snap_m``), on a network component that does not reach the demand network, or sharing a node with another site
  (``shared_node = "keep_first"``: lowest site ID kept) are excluded with the reason; ``shared_node = "error"`` stops
  instead.
- ``demand_nodes``: the network nodes of the demand locations. EXPERIMENTAL: a modelling assumption, not a set of
  developable sites.

Every source returns candidates with the columns the solvers read (``facility_id``, ``nearest_node`` on the facility
layer, ``site_name``) plus provenance. Nothing here decides that a site is physically or legally available.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .core import DISTRICT_NAME, PROJECTED_CRS

EXPERIMENTAL = "demand_node (experimental: network nodes of demand locations, buildability not assessed)"


class CandidateError(ValueError):
    """Candidate sites that cannot be used as given (the message names the sites)."""


def residential(
    graph: nx.MultiDiGraph,
    buildings: pd.DataFrame,
    basis: pd.DataFrame,
    snapping: pd.DataFrame,
    districts: gpd.GeoDataFrame,
    groups: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """The default pool: one candidate per network node of an eligible residential building.

    Only the node set matters, so ``basis`` (district totals, or weight 1 per district) only has to be positive.
    """
    from .assemble import on_facility_layer
    from .demand import allocate_children, build_demand_nodes
    from .demand_modes import AGES, has_split
    from .optimisation import build_candidate_pool

    columns = AGES if has_split(basis) else ("children_0_5",)
    alloc = allocate_children(buildings, basis, method="footprint", columns=columns)
    for column in set(AGES) - set(columns):  # no age split: the pool needs only the nodes
        alloc[column] = np.nan
    return on_facility_layer(
        build_candidate_pool(build_demand_nodes(graph, buildings, alloc, snapping, districts, groups))
    )


def demand_nodes(demand: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """EXPERIMENTAL: every demand-location node is a candidate (a modelling assumption, not developable sites)."""
    from .assemble import on_facility_layer
    from .optimisation import build_candidate_pool

    candidates = on_facility_layer(build_candidate_pool(demand))
    candidates["candidate_source"] = EXPERIMENTAL
    return candidates


def _node_points(graph: nx.MultiDiGraph, nodes: list[int]) -> gpd.GeoSeries:
    """Points of network nodes, in the CRS of the graph."""
    return gpd.GeoSeries(
        gpd.points_from_xy([graph.nodes[n]["x"] for n in nodes], [graph.nodes[n]["y"] for n in nodes]),
        crs=graph.graph.get("crs", "EPSG:4326"),
    )


def supplied(spec: dict, districts: gpd.GeoDataFrame, graph: nx.MultiDiGraph) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Supplied sites (``[candidates] source = "file"``).

    Returns (candidates, sites): ``sites`` lists every supplied site with its original ID, point, district, snapped node
    and distance, status and the reason for an exclusion.
    """
    from .assemble import on_facility_layer
    from .network import snap_buildings

    raw = gpd.read_file(spec["file"])
    if raw.crs is None:
        raise CandidateError(f"[candidates] {spec['file']}: no CRS defined")
    if spec["id_column"] not in raw.columns:
        raise CandidateError(f"[candidates] id column {spec['id_column']!r} not found (columns: {list(raw.columns)})")
    ids = raw[spec["id_column"]]
    if ids.isna().any() or ids.astype(str).duplicated().any():
        raise CandidateError(
            f"[candidates] {spec['id_column']}: missing or duplicate site IDs "
            f"{sorted(set(ids[ids.astype(str).duplicated()].astype(str)))[:5]}"
        )
    ids = ids.astype(str)
    empty = raw.geometry.isna() | raw.geometry.is_empty
    if empty.any():
        raise CandidateError(f"[candidates] sites without geometry or coordinates: {ids[empty].tolist()[:5]}")
    found = set(raw.geom_type)
    if not found <= {"Point", "Polygon", "MultiPolygon"}:
        raise CandidateError(
            f"[candidates] supported geometries are Point, Polygon, MultiPolygon; found {sorted(found)} "
            "(no geometry is converted automatically)"
        )
    invalid = ~raw.geometry.is_valid
    if invalid.any():
        raise CandidateError(f"[candidates] invalid geometries: {ids[invalid].tolist()[:5]}")
    shapes = raw.to_crs(PROJECTED_CRS).geometry.reset_index(drop=True)
    ids = ids.reset_index(drop=True)
    polygon = shapes.geom_type.isin(["Polygon", "MultiPolygon"])
    rule = spec.get("polygon_point", "")
    if polygon.any() and rule not in ("representative_point", "centroid"):
        raise CandidateError(
            "[candidates] the file has polygons: set polygon_point = 'representative_point' (a point "
            "inside each polygon) or 'centroid' (only where it lies inside the polygon)"
        )
    points = shapes.copy()
    if polygon.any():
        reduced = shapes[polygon].representative_point() if rule == "representative_point" else shapes[polygon].centroid
        outside = ~reduced.within(shapes[polygon])
        if outside.any():
            raise CandidateError(
                f"[candidates] polygon_point = 'centroid' lies outside polygons {ids[polygon][outside].tolist()[:5]}; "
                "use 'representative_point'"
            )
        points[polygon] = reduced
    sites = pd.DataFrame(
        {
            "site_id": ids,
            "x": points.x,
            "y": points.y,
            "geometry_type": shapes.geom_type,
            "status": "candidate",
            "reason": "",
        }
    )

    def exclude(mask: pd.Series, reason: str) -> None:
        mask = mask & (sites["status"] == "candidate")
        sites.loc[mask, "status"], sites.loc[mask, "reason"] = (
            "excluded",
            reason if isinstance(reason, str) else reason[mask],
        )

    column = spec.get("eligible_column", "")
    if column:
        if column not in raw.columns:
            raise CandidateError(f"[candidates] eligible_column {column!r} not found")
        value = raw[column].reset_index(drop=True)
        exclude(value.isna(), f"no value in {column} (eligibility not established)")
        allowed = {str(v) for v in spec.get("eligible_values", [])}
        exclude(
            ~value.astype(str).isin(allowed),
            "ineligible: " + column + " = " + value.astype(str) + f" not in {sorted(allowed)}",
        )
    frame = districts[["district_id", DISTRICT_NAME, "geometry"]].to_crs(PROJECTED_CRS)
    located = gpd.GeoDataFrame(
        sites[["site_id"]], geometry=gpd.points_from_xy(sites["x"], sites["y"]), crs=PROJECTED_CRS
    )
    hit = gpd.sjoin(located, frame, predicate="intersects", how="left").sort_values("district_id", kind="stable")
    sites["on_boundary"] = hit.index.value_counts().reindex(sites.index).gt(1)
    on_boundary = sites["on_boundary"] & (sites["status"] == "candidate")
    if on_boundary.any() and spec.get("boundary_site", "error") == "error":
        raise CandidateError(
            f"[candidates] sites exactly on a district boundary: {sites.loc[on_boundary, 'site_id'].tolist()[:5]}; "
            "move them into one district, or set boundary_site = 'lowest_id' to assign the lowest district ID"
        )
    sites["boundary_assignment"] = np.where(on_boundary, "lowest district ID (boundary_site = 'lowest_id')", "")
    hit = hit[~hit.index.duplicated()].reindex(sites.index)  # sorted: lowest district ID first
    sites["district_id"], sites[DISTRICT_NAME] = hit["district_id"], hit[DISTRICT_NAME]
    exclude(sites["district_id"].isna(), "outside every district")

    snapped = snap_buildings(
        graph,
        pd.DataFrame({"building_id": sites["site_id"], "rep_x": sites["x"], "rep_y": sites["y"]}),
    )
    sites["base_node"], sites["snap_distance_m"] = (
        snapped["nearest_node"].astype("int64"),
        snapped["snap_distance_m"],
    )
    limit = float(spec.get("max_snap_m", 100))
    exclude(
        sites["snap_distance_m"] > limit,
        f"more than max_snap_m = {limit:g} m from the walking network",
    )
    main = max(nx.strongly_connected_components(graph), key=len)
    exclude(
        ~sites["base_node"].isin(main),
        "on a network component that does not connect to the main walking network",
    )

    keep = sites[sites["status"] == "candidate"].sort_values("site_id", kind="stable")
    shared = keep[keep["base_node"].duplicated(keep=False)]
    if len(shared):
        groups = shared.groupby("base_node")["site_id"].apply(list).tolist()
        if spec.get("shared_node", "error") == "error":
            raise CandidateError(
                f"[candidates] sites snap to the same network node: {groups[:5]} "
                "(shared_node = 'keep_first' keeps the lowest site ID)"
            )
        first = keep.groupby("base_node")["site_id"].transform("first")
        drop = keep.index[keep["site_id"] != first]
        sites.loc[drop, "status"] = "excluded"
        sites.loc[drop, "reason"] = "same network node as site " + first[drop]
    chosen = sites[sites["status"] == "candidate"].sort_values("site_id", kind="stable")
    names = (
        raw[spec["name_column"]].reset_index(drop=True) if spec.get("name_column") in raw.columns else chosen["site_id"]
    )
    candidates = gpd.GeoDataFrame(
        {
            "demand_id": np.nan,
            DISTRICT_NAME: chosen[DISTRICT_NAME].to_numpy(),
            "nearest_node": chosen["base_node"].to_numpy(),
            "facility_id": ("site:" + chosen["site_id"]).to_numpy(),
            "facility_type": "candidate",
            "candidate_source": "supplied site (eligibility as given by the source)",
            "site_name": names.reindex(chosen.index).astype(str).to_numpy(),
            "union_origin": "supplied_site",
            "site_id": chosen["site_id"].to_numpy(),
            "snap_distance_m": chosen["snap_distance_m"].to_numpy(),
        },
        geometry=_node_points(graph, chosen["base_node"]).to_numpy(),
        crs=graph.graph.get("crs", "EPSG:4326"),
    )
    return on_facility_layer(candidates.reset_index(drop=True)), sites


def check_enough(candidates: gpd.GeoDataFrame, p: int, source: str) -> dict:
    """At least P distinct candidate nodes (before the relations and the screen)."""
    nodes = int(candidates["base_node"].nunique())
    if nodes < p:
        raise CandidateError(
            f"only {nodes} distinct candidate network node(s) ({len(candidates)} candidate(s), source "
            f"'{source}') for P = {p}: reduce P or supply more eligible sites"
        )
    return {"candidates": len(candidates), "distinct_nodes": nodes, "P": p}
