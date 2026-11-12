"""Demand modes: where the children aged 0-5 are.

Every mode returns the standard demand-origin table of :mod:`schema` (``origin_id``, ``district_id``,
``district_name``, ``x``, ``y``, ``children_0_5``, ``origin_type``, ``source_id``; ``age_0_2`` and ``age_3_5`` only
when the source has an age split, which is never invented). ``assemble.demand_locations`` turns it into demand
locations.

- ``census_hybrid`` (census-grid allocation, the default): district totals split by census-cell shares, footprint shares
  within cells; cells without a building become census-cell-centre origins (``demand.census_hybrid_allocation``).
- ``building_area`` (footprint allocation): district totals split by residential footprint shares
  (``demand.allocate_children``).
- ``direct``: supplied child counts at points (each point is an origin) or per polygon, allocated within the polygon to
  residential buildings by footprint (``allocate_children`` with the polygons as units) or in equal shares to the
  walk-network nodes inside it. Polygon centroids are never used. The network-node option is an EXPERIMENTAL assumption:
  node density reflects how paths and streets are mapped, not where children live, so these locations are allocation
  points, never observed child residences (label and template say so). A polygon crossing district boundaries is
  rejected or, with ``boundary_rule = "split"``, each part keeps the children allocated to its own buildings or nodes;
  polygon totals are conserved either way.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from . import schema
from .core import DISTRICT_NAME, PROJECTED_CRS

AGES = ("age_0_2", "age_3_5")
OVERLAP = 1e-9  # relative area below which a polygon-district or polygon-polygon overlap is ignored (rounding)


class DemandError(ValueError):
    """Supplied demand data that cannot be used as given (the message names the rows)."""


def has_split(frame: pd.DataFrame | None) -> bool:
    """Whether a table has both age columns (0-2 and 3-5), filled for every row."""
    return frame is not None and all(c in frame for c in AGES) and frame[list(AGES)].notna().all().all()


def describe(methods: dict) -> str:
    """The demand description of the build: count type, year, method (census year from the metadata if known)."""
    d = methods["demand"]
    head = f"{d['count_type']} {d['reference_year']} children aged 0-5"
    if d["mode"] == "census_hybrid":
        census = f"census {d['census_year']} cell shares" if d.get("census_year") else "census cell shares"
        return f"{head}, census-grid allocation ({census}, footprint within cells)"
    if d["mode"] == "building_area":
        return f"{head}, footprint allocation (footprint shares within districts)"
    if d["geometry"] == "points":
        return f"{head}, supplied points"
    if d["polygon_allocation"] == "building_area":
        return f"{head}, supplied polygons (footprint shares within polygons)"
    return f"{head}, supplied polygons (equal shares over network nodes within polygons; experimental assumption)"


def consistency(direct_totals: pd.Series, population: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Direct demand per district against the supplied district totals (``[population]``), never rescaled.

    ``status``: ``match`` (|difference| <= tolerance), ``differs`` (a numerical discrepancy), ``no_direct_demand``
    (the district has a supplied total > 0 but no direct demand: missing coverage), ``no_district_total`` (direct
    demand in a district the totals do not list).
    """
    supplied = population.set_index("district_id")["children_0_5"].astype(float)
    index = supplied.index.union(direct_totals.index)
    table = pd.DataFrame(
        {
            "population_total": supplied.reindex(index),
            "direct_total": direct_totals.reindex(index, fill_value=0.0),
        }
    )
    table["abs_difference"] = (table["direct_total"] - table["population_total"]).abs()
    table["pct_difference"] = (
        100
        * (table["direct_total"] - table["population_total"])
        / table["population_total"].where(table["population_total"] > 0)
    )
    table["status"] = np.select(
        [
            table["population_total"].isna(),
            (table["direct_total"] == 0) & (table["population_total"] > 0),
            table["abs_difference"] <= tolerance,
        ],
        ["no_district_total", "no_direct_demand", "match"],
        "differs",
    )
    return table.rename_axis("district_id").reset_index()


def _counts(frame: pd.DataFrame, spec: dict, what: str) -> pd.DataFrame:
    """``children_0_5`` (and the ages if configured) as numbers; missing, negative or inconsistent values raise."""
    columns = {"children_0_5": spec.get("children_0_5", "")} | {a: spec.get(a, "") for a in AGES}
    columns = {k: v for k, v in columns.items() if v}
    absent = [v for v in columns.values() if v not in frame.columns]
    if absent:
        raise DemandError(f"{what}: configured column(s) {absent} not found (columns: {list(frame.columns)})")
    out = pd.DataFrame({k: pd.to_numeric(frame[v], errors="coerce") for k, v in columns.items()}, index=frame.index)
    bad = out.index[out.isna().any(axis=1) | (out < 0).any(axis=1)]
    if len(bad):
        raise DemandError(f"{what}: {len(bad)} row(s) with missing, non-numeric or negative child counts")
    if set(AGES) <= set(out):
        total = out["age_0_2"] + out["age_3_5"]
        if "children_0_5" in out and ((total - out["children_0_5"]).abs() > schema.TOLERANCE).any():
            raise DemandError(f"{what}: age_0_2 + age_3_5 differs from children_0_5")
        out["children_0_5"] = total
    return out


def child_totals(population: dict, districts: pd.DataFrame, id_width: int) -> pd.DataFrame:
    """District child totals from a wide ``[population]`` table (one row per district)."""
    raw = pd.read_csv(
        population["file"],
        sep=population.get("separator", ","),
        encoding=population.get("encoding", "utf-8"),
        dtype={population["district_id_column"]: str},
    )
    counts = _counts(raw, population, "[population]")
    ids = raw[population["district_id_column"]].astype(str).str.strip()
    table = pd.concat(
        [pd.DataFrame({"district_id": ids.str.zfill(id_width) if id_width else ids}), counts],
        axis=1,
    )
    schema.check("child_totals", table, districts)
    return table


def census_hybrid(
    buildings: pd.DataFrame, children: pd.DataFrame, districts: pd.DataFrame, census_crs: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    """Census-grid allocation. Returns (origins, building demand, fallback cells, audit)."""
    from .demand import census_hybrid_allocation, load_census_children_cells

    cells = load_census_children_cells(districts)
    D, fallback, audit = census_hybrid_allocation(buildings, children, cells)
    centres = gpd.GeoSeries(gpd.points_from_xy(fallback["x_3035"], fallback["y_3035"]), crs=census_crs).to_crs(
        PROJECTED_CRS
    )
    names = districts.set_index("district_id")[DISTRICT_NAME]
    origins = pd.concat(
        [
            _from_buildings(buildings, D.to_numpy()),
            pd.DataFrame(
                {
                    "origin_id": "census_cell:" + fallback["cell_id"],
                    "district_id": fallback["district_id"].to_numpy(),
                    DISTRICT_NAME: names.loc[fallback["district_id"]].to_numpy(),
                    "x": centres.x.to_numpy(),
                    "y": centres.y.to_numpy(),
                    "children_0_5": fallback["children_0_5"].to_numpy(),
                    "origin_type": "census_cell_centre",
                    "source_id": fallback["cell_id"].astype(str).to_numpy(),
                    "footprint_m2": 0.0,
                    "excluded": False,
                }
            ),
        ],
        ignore_index=True,
    )
    return _ages(origins, children), D, fallback, audit


def building_area(buildings: pd.DataFrame, children: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Footprint allocation. Returns (origins, building demand)."""
    from .demand import allocate_children

    alloc = allocate_children(
        buildings,
        children,
        method="footprint",
        columns=AGES if has_split(children) else ("children_0_5",),
    )
    origins = _from_buildings(buildings, alloc["children_0_5"].to_numpy())
    if has_split(children):
        origins["age_0_2"], origins["age_3_5"] = (
            alloc["age_0_2"].to_numpy(),
            alloc["age_3_5"].to_numpy(),
        )
    return origins, alloc["children_0_5"]


def _from_buildings(buildings: pd.DataFrame, children: pd.Series) -> pd.DataFrame:
    """Demand origins from snapped buildings, with their allocated children."""
    return pd.DataFrame(
        {
            "origin_id": buildings["building_id"].to_numpy(),
            "district_id": buildings["district_id"].to_numpy(),
            DISTRICT_NAME: buildings[DISTRICT_NAME].to_numpy(),
            "x": buildings["rep_x"].to_numpy(),
            "y": buildings["rep_y"].to_numpy(),
            "children_0_5": children,
            "origin_type": "building",
            "source_id": buildings["building_id"].astype(str).to_numpy(),
            "footprint_m2": buildings["footprint_m2"].to_numpy(),
            "excluded": buildings["exclude_student_dormitory"].astype(bool).to_numpy(),
            "nearest_node": buildings["nearest_node"].to_numpy(),
            "snap_distance_m": buildings["snap_distance_m"].to_numpy(),
        }
    )


def _ages(origins: pd.DataFrame, children: pd.DataFrame) -> pd.DataFrame:
    """Census-grid allocation ages: each origin's children split by its district's age shares."""
    if has_split(children):
        totals = children.set_index("district_id")
        ratio = (totals["age_0_2"] / totals["children_0_5"]).reindex(origins["district_id"]).to_numpy()
        total = origins["children_0_5"].to_numpy()
        origins["age_0_2"], origins["age_3_5"] = total * ratio, total * (1 - ratio)
    return origins


# direct
def _read(spec: dict) -> gpd.GeoDataFrame:
    """Read the supplied demand file; refuse a file without coordinate system."""
    raw = gpd.read_file(spec["file"])
    if raw.crs is None:
        raise DemandError(f"[demand.direct] {spec['file']}: no CRS defined")
    ids = raw[spec["id_column"]] if spec["id_column"] in raw.columns else None
    if ids is None:
        raise DemandError(f"[demand.direct] id column {spec['id_column']!r} not found")
    if ids.isna().any() or ids.duplicated().any():
        raise DemandError(f"[demand.direct] {spec['id_column']}: missing or duplicate IDs")
    empty = raw.geometry.isna() | raw.geometry.is_empty
    if empty.any():
        raise DemandError(
            f"[demand.direct] {int(empty.sum())} row(s) without geometry: {ids[empty].astype(str).tolist()[:5]}"
        )
    allowed = {"points": {"Point"}, "polygons": {"Polygon", "MultiPolygon"}}[spec["geometry"]]
    found = set(raw.geom_type)
    if not found <= allowed:
        raise DemandError(
            f"[demand.direct] geometry = {spec['geometry']!r} allows {sorted(allowed)}, found {sorted(found)}; "
            "convert the data explicitly (no geometry is converted automatically)"
        )
    if spec["geometry"] == "polygons" and not raw.geometry.is_valid.all():
        raise DemandError(f"[demand.direct] invalid polygons: {ids[~raw.geometry.is_valid].astype(str).tolist()[:5]}")
    return raw.to_crs(PROJECTED_CRS).reset_index(drop=True)


def _locate(points: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> pd.DataFrame:
    """District of each point (strictly inside one district polygon), else NaN."""
    frame = districts[["district_id", DISTRICT_NAME, "geometry"]].to_crs(PROJECTED_CRS)
    hit = gpd.sjoin(points[["geometry"]], frame, predicate="within", how="left")
    if hit.index.duplicated().any():
        raise DemandError("a location lies inside two districts (overlapping district polygons)")
    return hit[["district_id", DISTRICT_NAME]]


def direct(
    spec: dict, districts: gpd.GeoDataFrame, graph: nx.MultiDiGraph, buildings: pd.DataFrame | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Supplied demand (``[demand.direct]``). Returns (origins, audit)."""
    raw = _read(spec)
    ids = raw[spec["id_column"]].astype(str)
    counts = _counts(raw, spec, "[demand.direct]")
    split = set(AGES) <= set(counts)
    given = spec.get("district_column") or None
    if given and given not in raw.columns:
        raise DemandError(f"[demand.direct] district_column {given!r} not found")
    if spec["geometry"] == "points":
        where = _locate(raw, districts)
        outside = where["district_id"].isna()
        if outside.any():
            raise DemandError(
                f"[demand.direct] {int(outside.sum())} point(s) outside every district (or on a boundary): "
                f"{ids[outside].tolist()[:5]}"
            )
        if given:
            _same_district(raw[given], where["district_id"], ids)
        origins = pd.DataFrame(
            {
                "origin_id": "point:" + ids,
                "district_id": where["district_id"].to_numpy(),
                DISTRICT_NAME: where[DISTRICT_NAME].to_numpy(),
                "x": raw.geometry.x,
                "y": raw.geometry.y,
                "children_0_5": counts["children_0_5"],
                "origin_type": "supplied_point",
                "source_id": ids,
            }
        )
        if split:
            origins["age_0_2"], origins["age_3_5"] = counts["age_0_2"], counts["age_3_5"]
        return origins, {
            "points": len(origins),
            "children_0_5": float(counts["children_0_5"].sum()),
        }
    return _polygons(spec, raw, ids, counts, split, given, districts, graph, buildings)


def _same_district(given: pd.Series, found: pd.Series, ids: pd.Series) -> None:
    """Stop if a supplied district differs from the district the location lies in."""
    wrong = given.astype(str).str.strip().to_numpy() != found.astype(str).to_numpy()
    if wrong.any():
        raise DemandError(
            f"[demand.direct] district_column disagrees with the location for {int(wrong.sum())} row(s): "
            f"{ids[wrong].tolist()[:5]}"
        )


def _polygons(
    spec: dict,
    raw: gpd.GeoDataFrame,
    ids: pd.Series,
    counts: pd.DataFrame,
    split: bool,
    given: pd.Series | None,
    districts: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    buildings: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Supplied demand polygons: each polygon's children go to the buildings (or the centre) inside it."""
    areas = raw.geometry.area
    overlap = gpd.sjoin(raw[["geometry"]], raw[["geometry"]], predicate="intersects")
    overlap = overlap[overlap.index < overlap["index_right"]]
    shared = [
        (i, j)
        for i, j in zip(overlap.index, overlap["index_right"], strict=True)
        if raw.geometry[i].intersection(raw.geometry[j]).area > OVERLAP * min(areas[i], areas[j])
    ]
    if shared:
        raise DemandError(
            f"[demand.direct] overlapping polygons (ambiguous allocation): {[(ids[i], ids[j]) for i, j in shared[:5]]}"
        )
    frame = districts[["district_id", "geometry"]].to_crs(PROJECTED_CRS)
    pieces = gpd.overlay(raw[["geometry"]].assign(row=raw.index), frame, how="intersection", keep_geom_type=True)
    pieces = pieces[pieces.geometry.area > OVERLAP * areas[pieces["row"]].to_numpy()]
    spread = pieces.groupby("row")["district_id"].nunique()
    crossing = spread.index[spread > 1]
    if len(crossing) and spec["boundary_rule"] == "reject":
        raise DemandError(
            f"[demand.direct] {len(crossing)} polygon(s) cross district boundaries (boundary_rule = 'reject'): "
            f"{ids[crossing].tolist()[:5]}"
        )
    if given:
        if len(crossing):
            raise DemandError(
                "[demand.direct] district_column is given but polygons cross district boundaries: "
                f"{ids[crossing].tolist()[:5]}"
            )
        located = pieces.drop_duplicates("row").set_index("row")["district_id"].reindex(raw.index)
        _same_district(raw[given], located.fillna(""), ids)

    if spec["polygon_allocation"] == "building_area":
        units = buildings.loc[~buildings["exclude_student_dormitory"].astype(bool)]
        points = gpd.GeoDataFrame(
            units[["building_id"]],
            geometry=gpd.points_from_xy(units["rep_x"], units["rep_y"]),
            crs=PROJECTED_CRS,
        )
        origin_type = "polygon_building"
    else:
        nodes = pd.DataFrame([(n, d["x"], d["y"]) for n, d in graph.nodes(data=True)], columns=["node", "lon", "lat"])
        xy = gpd.GeoSeries(
            gpd.points_from_xy(nodes["lon"], nodes["lat"]), crs=graph.graph.get("crs", "EPSG:4326")
        ).to_crs(PROJECTED_CRS)
        points = gpd.GeoDataFrame(nodes[["node"]], geometry=xy.to_numpy(), crs=PROJECTED_CRS)
        where = _locate(points, districts)
        points = points[where["district_id"].notna()].assign(**where[where["district_id"].notna()])
        origin_type = "polygon_node"
    inside = gpd.sjoin(points, raw[["geometry"]], predicate="within", how="inner").rename(
        columns={"index_right": "row"}
    )
    empty = sorted(set(raw.index) - set(inside["row"]))
    if empty:
        what = "eligible residential buildings" if origin_type == "polygon_building" else "walk-network nodes"
        raise DemandError(f"[demand.direct] {len(empty)} polygon(s) contain no {what}: {ids[empty].tolist()[:5]}")

    columns = list(AGES) if split else ["children_0_5"]
    if origin_type == "polygon_building":
        from .demand import allocate_children

        units = buildings.loc[inside.index].assign(district_id=inside["row"].to_numpy())  # polygons as allocation units
        totals = counts.assign(district_id=raw.index)
        alloc = allocate_children(units.reset_index(drop=True), totals, method="footprint", columns=tuple(columns))
        rows = buildings.loc[inside.index]
        origins = pd.DataFrame(
            {
                "origin_id": "polygon:"
                + ids[inside["row"]].to_numpy()
                + ":building:"
                + rows["building_id"].astype(str).to_numpy(),
                "district_id": rows["district_id"].to_numpy(),
                DISTRICT_NAME: rows[DISTRICT_NAME].to_numpy(),
                "x": rows["rep_x"].to_numpy(),
                "y": rows["rep_y"].to_numpy(),
                "children_0_5": alloc["children_0_5"].to_numpy(),
                "origin_type": origin_type,
                "source_id": ids[inside["row"]].to_numpy(),
                "footprint_m2": rows["footprint_m2"].to_numpy(),
                "nearest_node": rows["nearest_node"].to_numpy(),
                "snap_distance_m": rows["snap_distance_m"].to_numpy(),
            }
        )
        for column in AGES if split else ():
            origins[column] = alloc[column].to_numpy()
    else:
        share = 1.0 / inside.groupby("row")["node"].transform("size")
        origins = pd.DataFrame(
            {
                "origin_id": "polygon:"
                + ids[inside["row"]].to_numpy()
                + ":node:"
                + inside["node"].astype(str).to_numpy(),
                "district_id": inside["district_id"].to_numpy(),
                DISTRICT_NAME: inside[DISTRICT_NAME].to_numpy(),
                "x": inside.geometry.x.to_numpy(),
                "y": inside.geometry.y.to_numpy(),
                "children_0_5": (share * counts["children_0_5"].reindex(inside["row"]).to_numpy()).to_numpy(),
                "origin_type": origin_type,
                "source_id": ids[inside["row"]].to_numpy(),
                "nearest_node": inside["node"].to_numpy(),
                "snap_distance_m": 0.0,
            }
        )
        for column in AGES if split else ():
            origins[column] = (share * counts[column].reindex(inside["row"]).to_numpy()).to_numpy()
    origins = origins.sort_values("origin_id", kind="stable").reset_index(drop=True)
    gap = (origins.groupby("source_id")["children_0_5"].sum().reindex(ids) - counts["children_0_5"].set_axis(ids)).abs()
    if gap.max() > schema.TOLERANCE * max(1.0, float(counts["children_0_5"].max())):
        raise DemandError(f"[demand.direct] polygon totals not conserved (max error {gap.max():.3g})")
    return origins, {
        "polygons": len(raw),
        "crossing_district_boundaries": len(crossing),
        "origins": len(origins),
        "children_0_5": float(counts["children_0_5"].sum()),
        "max_polygon_error": float(gap.max()),
    }
