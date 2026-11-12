"""Pedestrian network with explicit legal-access rules, built from a saved OSM snapshot.

OSMnx's ``walk`` filter keeps ways on which walking is prohibited (``motorroad=yes``, ``access=no``,
``foot=use_sidepath``, ``busway``) and ignores barrier nodes. This module classifies every cached way and
barrier before OSMnx builds the graph, so OSMnx's own construction (buffered truncation, largest
component, simplification, lengths) is otherwise unchanged. Precedence follows the OSM access scheme:

    foot=*  >  access=*  >  implied defaults of the way (motorroad, busway)

Classes
    public       walkable by anyone;
    destination  walkable only by people with business there (private, permit, delivery, employees;
                 private or locked gates): allowed at the start or the end of a trip, never as a through route;
    excluded     not walkable (removed).

Routing graph. Destination-only links are routed with a three-layer graph: a trip is R* P* R*, where P are
links of the main public network and R all other kept links (destination-only links, gate passages and
public links that are reachable only through them). Layer A (origins) holds R links, layer B the P links,
layer C (facilities) R links; zero-cost switches lead A -> B -> C at every node. Demand locations keep their
OSM node ids (layer A); facilities and candidate sites use their layer-C id. With ``rules=False`` the
filter is the identity and the original OSMnx graph is reproduced exactly (checked by the build script).
"""

from __future__ import annotations

import contextlib
import copy
import functools
from collections import Counter
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import networkx as nx
import osmnx as ox

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd

from . import settings as p26  # city settings (CONFIG, PROJECTED_CRS); earlier: primary_2025_26
from .core import WALK_NETWORK_TYPE, WALKING_SPEED_M_PER_MIN, get_bbox_wgs84
from .network import osm_cache_only

VERSION = "pedestrian_access_v1"
VERSION_FULL = "pedestrian_access_v2"  # v1 rules on the full highway snapshot (cycleways, private ways, sidewalks)
VERSION_FINAL = "pedestrian_access_v3"  # v2 with sidewalk-obligation roads kept as sidewalk proxies (final)
RULE_SET = p26.CONFIG.get("osm.network_rule_set")  # rule set named by the city configuration
if RULE_SET not in (VERSION, VERSION_FULL, VERSION_FINAL):
    raise p26.ConfigError(
        f"{p26.CONFIG.source}: [osm] network_rule_set {RULE_SET!r} is not one of "
        f"{[VERSION, VERSION_FULL, VERSION_FINAL]}"
    )
SIDEWALK_KEYS = ("sidewalk", "sidewalk:both", "sidewalk:left", "sidewalk:right")
LAYER_B, LAYER_C = 2**52, 2**53  # node-id offsets of the public and the facility layer

PUBLIC = {"yes", "designated", "permissive", "destination", "customers", "visitors", "unknown"}
DESTINATION = {"private", "permit", "delivery", "employees"}
EXCLUDED = {"no", "use_sidepath", "bus", "emergency", "discouraged"}
# Field and forest tracks: access=agricultural/forestry signs restrict vehicles; walking in open landscape
# and forest is permitted in NRW (LNatSchG NRW §57, LFoG NRW §2).
LANDSCAPE_WAYS = {"track", "path", "bridleway"}
LANDSCAPE_VEHICLE_ONLY = {"agricultural", "forestry"}
# Barriers a pedestrian passes; their access tags concern vehicles unless foot=* says otherwise.
PASSABLE_BARRIERS = {
    "bollard",
    "block",
    "kerb",
    "cycle_barrier",
    "log",
    "height_restrictor",
    "parking_lane",
    "tree",
    "bump_gate",
    "lift_gate",
}


def _value_class(value: str) -> str:
    """Class (public, destination, excluded) of an access tag value."""
    if value in PUBLIC:
        return "public"
    if value in DESTINATION:
        return "destination"
    if value in EXCLUDED:
        return "excluded"
    return "public"  # unknown value: keep, reported by the audit


def way_class(tags: dict) -> tuple[str, str]:
    """(public | destination | excluded, reason) for one OSM way of the cached walk query."""
    foot, access, highway = tags.get("foot"), tags.get("access"), tags.get("highway")
    if foot is not None:
        return _value_class(foot), f"foot={foot}"
    if access is not None and access not in PUBLIC:
        if access in LANDSCAPE_VEHICLE_ONLY and highway in LANDSCAPE_WAYS:
            return "public", f"access={access} on {highway} (vehicle restriction)"
        if access in LANDSCAPE_VEHICLE_ONLY:
            return "destination", f"access={access}"
        return _value_class(access), f"access={access}"
    if tags.get("motorroad") == "yes":
        return "excluded", "motorroad=yes"
    if highway == "busway":
        return "excluded", "highway=busway"
    if highway == "cycleway":  # cycle path (Z. 237): pedestrians only where foot=* allows them
        return "excluded", "highway=cycleway without foot=*"
    if any(tags.get(k) == "separate" for k in SIDEWALK_KEYS):
        return (
            "excluded",
            "sidewalk=separate",
        )  # pedestrians use the separately mapped sidewalk (StVO §25)
    if tags.get("service") == "private" and access is None:
        return "destination", "service=private"
    return "public", "default" if access is None else f"access={access}"


def way_class_sidewalk_proxy(tags: dict) -> tuple[str, str]:
    """Final rule set (pedestrian_access_v3): ``way_class`` plus roads as sidewalk proxies.

    A road whose pedestrians must use a parallel sidewalk (foot=use_sidepath, sidewalk=separate) stays in the network
    as a proxy for that sidewalk. The obligation concerns the surface, not the corridor; crossing the road is legal
    (StVO §25(3)), and OSM maps crossings and sidewalk continuity incompletely (see PEDESTRIAN_NETWORK.md).
    """
    c, why = way_class(tags)
    if c == "excluded" and why in ("foot=use_sidepath", "sidewalk=separate"):
        return "public", f"{why} (sidewalk proxy)"
    return c, why


def barrier_class(tags: dict) -> tuple[str, str]:
    """(pass | destination | excluded, reason) for a node carrying barrier=*."""
    barrier = tags.get("barrier")
    if barrier is None:
        return "pass", ""
    foot, access, locked = tags.get("foot"), tags.get("access"), tags.get("locked")
    if foot is not None:
        c = _value_class(foot)
        return ("pass" if c == "public" else c), f"barrier={barrier}, foot={foot}"
    if barrier in PASSABLE_BARRIERS:
        return "pass", ""
    if access == "no":
        return "excluded", f"barrier={barrier}, access=no"
    if access in DESTINATION or locked == "yes":
        return "destination", f"barrier={barrier}, " + (f"access={access}" if access in DESTINATION else "locked=yes")
    return "pass", ""


def filter_response(
    response: dict,
    rules: bool = True,
    way_rule: Callable[[dict], tuple[str, str]] = way_class,
    barrier_rule: Callable[[dict], tuple[str, str]] = barrier_class,
) -> tuple[dict, dict]:
    """Classify ways and barriers of one Overpass response; returns (response for OSMnx, audit)."""
    if not rules:
        return response, {"rules": False}
    elements = response.get("elements", [])
    nodes = {e["id"]: e for e in elements if e["type"] == "node"}
    ways = [e for e in elements if e["type"] == "way"]
    removed, cls_count, keep = {}, Counter(), []
    proxies = set(sidepath_parallel_check(response)["way_ids"]) if way_rule is way_class else set()
    for w in ways:
        c, why = way_rule(w.get("tags", {}))
        if c == "excluded" and w["id"] in proxies:  # mandatory sidepath/sidewalk missing from the data: keep the road
            c, why = "public", f"{why}, kept: no parallel walkable way"
        if c == "excluded":
            removed[w["id"]] = why
            continue
        keep.append({**w, "tags": {**w.get("tags", {}), "ped_access": c}})
        cls_count[(c, why)] += 1
    used = {n for w in keep for n in w["nodes"]}
    barriers = {}
    for n in used:
        c, why = barrier_rule(nodes[n].get("tags", {}))
        if c != "pass":
            barriers[n] = (c, why)

    # Split every way at a non-passable barrier: each way end touching it gets its own copy of the node.
    # Destination-only barriers get a short connector (a destination-only link) between the copies.
    new_ways, next_node, next_way, origin_of = [], -1, -1, {}
    for w in keep:
        if not any(n in barriers for n in w["nodes"]):
            new_ways.append(w)
            continue
        parts, part = [], []
        for n in w["nodes"]:
            if n not in barriers:
                part.append(n)
                continue
            if part:  # close the current part at its own copy of the barrier
                part.append(next_node)
                origin_of[next_node] = n
                next_node -= 1
                parts.append(part)
            part = [next_node]  # the next part starts at another copy
            origin_of[next_node] = n
            next_node -= 1
        parts.append(part)
        for k, p in enumerate(p for p in parts if len(p) > 1):
            new_ways.append(
                {
                    **w,
                    "id": w["id"] if k == 0 else next_way,
                    "nodes": p,
                    "tags": {**w["tags"], "split_from_way": str(w["id"])},
                }
            )
            next_way -= k > 0
    used_copies = {n for w in new_ways for n in w["nodes"] if n < 0 and n in origin_of}
    copies: dict[int, list[int]] = {}
    for a in sorted(used_copies, reverse=True):
        copies.setdefault(origin_of[a], []).append(a)
    new_nodes = [
        {k: v for k, v in nodes[origin_of[a]].items() if k != "tags"} | {"id": a}
        for a in sorted(used_copies, reverse=True)
    ]
    for n, (c, _) in barriers.items():
        if c == "destination":  # connect all copies of the barrier through destination-only links
            cs = copies.get(n, [])
            for a, b in zip(cs, cs[1:]):
                new_ways.append(
                    {
                        "type": "way",
                        "id": next_way,
                        "nodes": [a, b],
                        "tags": {
                            "highway": "footway",
                            "ped_access": "destination",
                            "barrier_passage": str(n),
                        },
                    }
                )
                next_way -= 1
    out = copy.deepcopy({k: v for k, v in response.items() if k != "elements"})
    out["elements"] = [e for e in elements if e["type"] == "node"] + new_nodes + new_ways
    audit = {
        "ways_in": len(ways),
        "ways_excluded": len(removed),
        "excluded_by_reason": dict(Counter(removed.values())),
        "kept_by_class_and_reason": {f"{c} | {why}": n for (c, why), n in sorted(cls_count.items())},
        "barriers_non_passable": len(barriers),
        "barriers_by_class_and_reason": dict(Counter(f"{c} | {why}" for c, why in barriers.values())),
        "excluded_way_ids": sorted(removed),
        "barrier_node_ids": {
            c: sorted(n for n, (cc, _) in barriers.items() if cc == c) for c in ("destination", "excluded")
        },
    }
    return out, audit


def sidepath_parallel_check(response: dict, distance_m: float = 25.0) -> dict:
    """For foot=use_sidepath ways: is a walkable way present alongside (within ``distance_m``, >= half the length)?"""
    import geopandas as gpd
    import shapely

    elements = response.get("elements", [])
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}
    ways = [e for e in elements if e["type"] == "way" and len(e.get("nodes", [])) > 1]
    walkable = [w for w in ways if way_class(w.get("tags", {}))[0] != "excluded"]
    side = [
        w
        for w in ways
        if w.get("tags", {}).get("foot") == "use_sidepath" or (way_class(w.get("tags", {}))[1] == "sidewalk=separate")
    ]
    line = lambda w: shapely.LineString([nodes[n] for n in w["nodes"]])
    wk = gpd.GeoSeries([line(w) for w in walkable], crs=4326).to_crs(p26.PROJECTED_CRS)
    sd = gpd.GeoSeries([line(w) for w in side], crs=4326).to_crs(p26.PROJECTED_CRS)
    tree = shapely.STRtree(wk.values)
    without = []
    for w, g in zip(side, sd.values):
        buf = g.buffer(distance_m)
        near = tree.query(buf, predicate="intersects")
        cover = shapely.union_all([wk.values[k].intersection(buf) for k in near]).length if len(near) else 0.0
        if cover < 0.5 * g.length:
            without.append(w["id"])
    return {
        "use_sidepath_or_separate_sidewalk_ways": len(side),
        "without_parallel_walkable_way": len(without),
        "way_ids": without,
    }


@contextlib.contextmanager
def filtered_overpass(
    rules: bool = True, audits: list | None = None, response_path: str | Path | None = None, **rule_kw: Any
) -> Iterator[None]:
    """Make OSMnx read the cached walk query (or a saved Overpass response) through ``filter_response``."""
    import json
    from pathlib import Path

    from osmnx import _overpass
    from osmnx import graph as ox_graph

    original, simplify = (
        _overpass._download_overpass_network,
        ox_graph.simplification.simplify_graph,
    )
    source = (
        (lambda *a, **k: iter([json.loads(Path(response_path).read_text(encoding="utf-8"))]))
        if response_path
        else original
    )

    def patched(*args: Any, **kwargs: Any) -> Iterator[dict]:
        """Yield the filtered responses (and record their audits)."""
        for response in source(*args, **kwargs):
            out, audit = filter_response(response, rules, **rule_kw)
            if rules:
                audit["sidepath"] = sidepath_parallel_check(response)
            if audits is not None:
                audits.append(audit)
            yield out

    tags_way = list(ox.settings.useful_tags_way)
    _overpass._download_overpass_network = patched
    if rules:
        ox_graph.simplification.simplify_graph = functools.partial(simplify, edge_attrs_differ=["ped_access"])
        ox.settings.useful_tags_way = tags_way + [
            "ped_access",
            "barrier_passage",
            "access",
            "foot",
            "motorroad",
        ]
    try:
        yield
    finally:
        _overpass._download_overpass_network = original
        ox_graph.simplification.simplify_graph = simplify
        ox.settings.useful_tags_way = tags_way


def load_pedestrian_graph(
    districts: gpd.GeoDataFrame, rules: bool = True, response_path: str | Path | None = None, **rule_kw: Any
) -> tuple[nx.MultiDiGraph, dict]:
    """Base walking graph (OSM node ids) from the OSM cache with the access rules; returns (graph, audit)."""
    audits: list = []
    with osm_cache_only(), filtered_overpass(rules, audits, response_path, **rule_kw):
        graph = ox.graph_from_bbox(bbox=get_bbox_wgs84(districts), network_type=WALK_NETWORK_TYPE)
    if rules:
        _restore_ways(graph)
    for *_, data in graph.edges(keys=True, data=True):
        data["walk_time_min"] = float(data.get("length", 0.0)) / WALKING_SPEED_M_PER_MIN
    graph.graph["pedestrian_network"] = (
        (VERSION_FULL if response_path else VERSION) if rules else "osmnx_walk_unfiltered"
    )
    if len(audits) != 1:
        raise RuntimeError(f"expected one cached Overpass response, got {len(audits)}")
    return graph, audits[0]


def _restore_ways(graph: nx.MultiDiGraph) -> None:
    """ped_access as one value per edge (links of different classes are never merged)."""
    for *_, d in graph.edges(keys=True, data=True):
        v = d.get("ped_access", "public")
        if isinstance(v, list):
            if len(set(v)) != 1:
                raise RuntimeError("an edge merges links of different access classes")
            v = v[0]
        d["ped_access"] = v


def link_classes(graph: nx.MultiDiGraph) -> dict:
    """P (main public network) or R (destination-only or reachable only through such links) per edge."""
    public = nx.MultiDiGraph()
    public.add_edges_from((u, v, k) for u, v, k, d in graph.edges(keys=True, data=True) if d["ped_access"] == "public")
    main = max(nx.weakly_connected_components(public), key=len) if public.number_of_nodes() else set()
    return {
        (u, v, k): ("P" if d["ped_access"] == "public" and u in main and v in main else "R")
        for u, v, k, d in graph.edges(keys=True, data=True)
    }


def layered_graph(graph: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, Counter]:
    """Routing graph for R* P* R* trips (see module docstring). Node attributes are copied to every layer."""
    cls = link_classes(graph)
    H = nx.MultiDiGraph(**graph.graph)
    H.graph["layers"] = {
        "A": "origins (OSM ids)",
        "B": f"public (+{LAYER_B})",
        "C": f"facilities (+{LAYER_C})",
    }
    for n, d in graph.nodes(data=True):
        for off, layer in ((0, "A"), (LAYER_B, "B"), (LAYER_C, "C")):
            H.add_node(n + off, **d, layer=layer, base_node=n)
        H.add_edge(n, n + LAYER_B, length=0.0, walk_time_min=0.0, layer_switch="A-B")
        H.add_edge(n + LAYER_B, n + LAYER_C, length=0.0, walk_time_min=0.0, layer_switch="B-C")
    for (u, v, k), c in cls.items():
        d = graph.edges[u, v, k]
        if c == "P":
            H.add_edge(u + LAYER_B, v + LAYER_B, **d, link_class="P")
        else:
            H.add_edge(u, v, **d, link_class="R")
            H.add_edge(u + LAYER_C, v + LAYER_C, **d, link_class="R")
    return H, Counter(cls.values())


def facility_node(base_node: int) -> int:
    """Node id of a facility on the facility layer (layer C)."""
    return int(base_node) + LAYER_C


def base_node(node: int) -> int:
    """Base OSM node id of a node on any layer."""
    node = int(node)
    return node - LAYER_C if node >= LAYER_C else (node - LAYER_B if node >= LAYER_B else node)
