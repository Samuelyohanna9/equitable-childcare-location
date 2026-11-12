"""Demand estimation: districts, forecast, residential buildings, dormitories, census grid and allocation.

The projected children of each district (forecast) are allocated to residential buildings and census-cell
fallback origins (census-grid allocation, :func:`census_hybrid_allocation`; footprint allocation and the other
weightings,
:func:`allocate_children`) and aggregated to demand locations on the network (:func:`build_demand_nodes`).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import networkx as nx

from .core import (
    DISTRICT_NAME,
    PROJECTED_CRS,
    RESIDENTIAL_BUILDING_TYPES,
    WGS84,
    attach_vulnerability_groups_to_demand,
    get_bbox_wgs84,
    load_district_data,
)
from .network import osm_cache_only
from .settings import (
    ALKIS_GPKG,
    ALKIS_LAYER,
    ALKIS_MIXED_CODE,
    ALKIS_RESIDENTIAL_CODES,
    ALLOCATION_METHODS,
    BUILDINGS,
    CENSUS,
    CENSUS_CELL_M,
    CENSUS_CRS,
    CENSUS_ZIP,
    DISTRICT_ID_COLUMN,
    DISTRICT_ID_WIDTH,
    DISTRICT_NAME_COLUMN,
    DISTRICTS_FILE,
    DORM_EXCLUDE_MIN_SHARE,
    DORM_REPORT_MIN_SHARE,
    EXPECTED_DISTRICT_COUNT,
    FORECAST,
    FORECAST_AGE_GROUPS,
    FORECAST_FILE,
    FORECAST_YEAR,
    NON_STUDENT_SIGNAL,
    PRIMARY_ALLOCATION,
    STUDENT_DECISIONS_FILE,
    STUDENT_DECISIONS_SHA256,
    STUDENT_SIGNAL,
)
from .validation import sha256_file

KEY = ["district_id", "gx", "gy"]


def _district_code(raum: pd.Series, pattern: str = FORECAST["district_code_pattern"]) -> pd.Series:
    """District code from an area label such as '23 Bahnhof' (group 1 of the configured pattern)."""
    return raum.astype(str).str.extract(pattern)[0]


def _check_district_count(found: int, what: str, expected: int | None = EXPECTED_DISTRICT_COUNT) -> None:
    """Stop if a table does not have the expected number of districts."""
    if expected is not None and found != expected:
        raise ValueError(f"Expected {expected} {what}, found {found} ([validation] district_count).")


def load_districts(path: str | Path = DISTRICTS_FILE) -> gpd.GeoDataFrame:
    """The district polygons with their IDs and names."""
    districts = load_district_data(str(path), required_columns=[DISTRICT_NAME_COLUMN, DISTRICT_ID_COLUMN])
    districts = districts.rename(columns={DISTRICT_ID_COLUMN: "district_id", DISTRICT_NAME_COLUMN: DISTRICT_NAME})
    districts["district_id"] = districts["district_id"].astype(str).str.zfill(DISTRICT_ID_WIDTH)
    if not districts["district_id"].is_unique:
        raise ValueError("District IDs are not unique.")
    _check_district_count(len(districts), "districts")
    return districts


def load_forecast_children(
    path: str | Path = FORECAST_FILE,
    year: int = FORECAST_YEAR,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Projected children aged 0-2, 3-5 and 0-5 for every district.

    Returns the district table and the citywide row, which is kept separate
    because the forecast rounds every cell independently.
    """
    f = FORECAST
    area, age, value = f["area_column"], f["age_column"], f["value_column"]
    forecast = pd.read_csv(path, sep=f["separator"], encoding=f["encoding"])
    forecast = forecast[(forecast[f["year_column"]] == year) & (forecast[f["sex_column"]] == f["all_sexes_value"])]
    forecast = forecast[forecast[age].isin(FORECAST_AGE_GROUPS.values())].copy()
    if forecast[value].isna().any() or (forecast[value] < 0).any():
        raise ValueError("Forecast child counts must be complete and non-negative.")
    forecast["district_id"] = _district_code(forecast[area], f["district_code_pattern"])

    city = forecast[forecast[area].str.contains(f["city_total_marker"], na=False)]
    city_values = {key: int(city.loc[city[age].eq(label), value].sum()) for key, label in FORECAST_AGE_GROUPS.items()}
    city_values["children_0_5"] = sum(city_values.values())

    districts = forecast[forecast["district_id"].notna()]
    table = districts.pivot_table(index="district_id", columns=age, values=value, aggfunc="sum").rename(
        columns={label: key for key, label in FORECAST_AGE_GROUPS.items()}
    )
    duplicates = districts.duplicated(subset=["district_id", age]).sum()
    if duplicates:
        raise ValueError(f"Forecast has {duplicates} duplicate district-age rows.")
    table = table[list(FORECAST_AGE_GROUPS)].astype(int)
    table["children_0_5"] = table.sum(axis=1)
    table = table.reset_index()
    _check_district_count(len(table), "forecast districts")
    return table, city_values


def load_alkis_residential_buildings(districts: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """ALKIS residential and mixed-use buildings assigned to districts.

    Building parts (``Bauteil``) and other structures are excluded so that no
    footprint is counted twice. Each building is assigned to the district that
    contains its representative point.
    """
    b = BUILDINGS
    function, kind, oid = b["function_column"], b["object_type_column"], b["id_column"]
    raw = gpd.read_file(ALKIS_GPKG, layer=ALKIS_LAYER).to_crs(PROJECTED_CRS)
    audit: dict[str, object] = {"alkis_features": int(len(raw))}
    eligible = raw[raw[function].isin(ALKIS_RESIDENTIAL_CODES)]
    audit["residential_code_features"] = int(len(eligible))
    audit["residential_code_by_gebnutzbez"] = eligible[kind].value_counts().to_dict()
    eligible = eligible[eligible[kind].eq(b["building_value"])].copy()
    audit["excluded_bauteil_features_total"] = int(raw[kind].eq(b["part_value"]).sum())
    empty = eligible.geometry.isna() | eligible.geometry.is_empty
    audit["empty_geometries"] = int(empty.sum())
    eligible = eligible[~empty]
    invalid = ~eligible.geometry.is_valid
    audit["invalid_geometries_repaired"] = int(invalid.sum())
    if invalid.any():
        eligible.loc[invalid, "geometry"] = eligible.loc[invalid, "geometry"].make_valid()
    audit["duplicate_oid"] = int(eligible[oid].duplicated().sum())
    audit["duplicate_geometry"] = int(eligible.geometry.normalize().to_wkb().duplicated().sum())
    if audit["duplicate_oid"] or audit["duplicate_geometry"]:
        raise ValueError(f"Duplicate ALKIS residential buildings: {audit}")

    eligible["footprint_m2"] = eligible.geometry.area
    eligible["is_mixed_use"] = eligible[function].eq(ALKIS_MIXED_CODE)
    points = eligible[[oid]].copy()
    points["geometry"] = eligible.geometry.representative_point()
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=PROJECTED_CRS)
    district_frame = districts[["district_id", DISTRICT_NAME, "geometry"]].to_crs(PROJECTED_CRS)
    joined = gpd.sjoin(points, district_frame, predicate="within", how="left")
    if joined.index.duplicated().any():
        raise ValueError("A building representative point fell inside two districts.")
    eligible["district_id"] = joined["district_id"]
    eligible[DISTRICT_NAME] = joined[DISTRICT_NAME]
    outside = eligible["district_id"].isna()
    audit["outside_districts_dropped"] = int(outside.sum())
    eligible = eligible[~outside].copy()

    # Buildings whose footprint crosses a district boundary keep the district of
    # their representative point; the count is reported.
    crossing = gpd.sjoin(eligible[["geometry"]], district_frame[["geometry"]], predicate="intersects")
    audit["footprints_touching_multiple_districts"] = int(crossing.index.value_counts().gt(1).sum())
    eligible["rep_x"] = points.loc[eligible.index].geometry.x
    eligible["rep_y"] = points.loc[eligible.index].geometry.y
    eligible = eligible.reset_index(drop=True)
    eligible["building_id"] = eligible[oid]
    audit["eligible_buildings"] = int(len(eligible))
    audit["eligible_mixed_use"] = int(eligible["is_mixed_use"].sum())
    audit["footprint_total_m2"] = float(eligible["footprint_m2"].sum())
    if "aktualit" in eligible:  # ALKIS update date (optional in other building sources)
        audit["aktualit_min"] = str(eligible["aktualit"].min())
        audit["aktualit_max"] = str(eligible["aktualit"].max())
    # The building function code is kept under its source name; ALKIS descriptive columns are kept when present.
    keep = [
        "building_id",
        function,
        "funktion",
        "is_mixed_use",
        "aktualit",
        "name",
        "footprint_m2",
        "district_id",
        DISTRICT_NAME,
        "rep_x",
        "rep_y",
        "geometry",
    ]
    return eligible[[c for c in keep if c in eligible.columns]], audit


def load_osm_residential_tags(districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """OSM residential building polygons with the tags used for dormitory and storey matching."""
    import osmnx as ox

    with osm_cache_only():
        raw = ox.features_from_bbox(
            bbox=get_bbox_wgs84(districts),
            tags={"building": RESIDENTIAL_BUILDING_TYPES},
        )
    raw = raw.to_crs(PROJECTED_CRS)
    raw = raw[raw.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index()
    columns = [
        "element",
        "id",
        "building",
        "name",
        "operator",
        "residential",
        "building:levels",
        "geometry",
    ]
    for column in columns:
        if column not in raw.columns:
            raw[column] = pd.NA
    osm = raw[columns].copy()
    osm["osm_ref"] = osm["element"].astype(str) + "/" + osm["id"].astype(str)
    levels = pd.to_numeric(osm["building:levels"], errors="coerce")
    osm["levels"] = levels.where((levels >= 1) & (levels <= 30))
    text = osm[["name", "operator", "residential"]].astype("string").fillna("").agg(" ".join, axis=1)
    osm["student_signal"] = (
        text.str.contains(STUDENT_SIGNAL) | osm["residential"].astype("string").eq("university").fillna(False)
    ).astype(bool)
    osm["non_student_signal"] = text.str.contains(NON_STUDENT_SIGNAL).astype(bool)
    return gpd.GeoDataFrame(osm, geometry="geometry", crs=PROJECTED_CRS)


def classify_dormitories(
    buildings: gpd.GeoDataFrame,
    osm: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Flag ALKIS buildings that are clearly student-only dormitories.

    A building is excluded only when all three conditions hold:

    1. it overlaps OSM ``building=dormitory`` polygons with an explicit student
       signal in ``name``, ``operator`` or ``residential=university``, and no
       family, senior or care signal;
    2. at least ``DORM_EXCLUDE_MIN_SHARE`` of its ALKIS footprint lies inside
       those polygons;
    3. its ALKIS function is purely residential (not mixed use).

    Every other overlap of at least ``DORM_REPORT_MIN_SHARE`` with any OSM
    dormitory or student-signalled building is listed as uncertain and kept.
    """
    candidates = osm[(osm["building"].eq("dormitory")) | osm["student_signal"]].copy()
    candidates["osm_class"] = np.select(
        [
            candidates["building"].eq("dormitory") & candidates["student_signal"] & ~candidates["non_student_signal"],
            candidates["building"].eq("dormitory"),
        ],
        ["student_dormitory", "dormitory_unspecified"],
        default="student_signal_non_dormitory",
    )
    pairs = gpd.sjoin(
        buildings[["building_id", "geometry"]],
        candidates[["osm_ref", "osm_class", "name", "operator", "geometry"]],
        predicate="intersects",
    )
    rows = []
    for index, row in pairs.iterrows():
        building_geom = buildings.geometry.loc[index]
        osm_geom = candidates.geometry.loc[row["index_right"]]
        overlap = building_geom.intersection(osm_geom).area
        rows.append(
            {
                "building_index": index,
                "building_id": row["building_id"],
                "osm_ref": row["osm_ref"],
                "osm_class": row["osm_class"],
                "osm_name": row["name"],
                "osm_operator": row["operator"],
                "overlap_m2": overlap,
            }
        )
    matches = pd.DataFrame(rows)
    buildings = buildings.copy()
    buildings["student_dorm_share"] = 0.0
    buildings["exclude_student_dormitory"] = False
    if matches.empty:
        return buildings, matches
    matches["footprint_m2"] = buildings.loc[matches["building_index"], "footprint_m2"].to_numpy()
    matches["is_mixed_use"] = buildings.loc[matches["building_index"], "is_mixed_use"].to_numpy()
    student_share = (
        (
            matches[matches["osm_class"].eq("student_dormitory")].groupby("building_index")["overlap_m2"].sum()
            / buildings["footprint_m2"]
        )
        .dropna()
        .clip(upper=1.0)
    )
    any_share = matches.groupby("building_index")["overlap_m2"].sum() / buildings["footprint_m2"]
    buildings.loc[student_share.index, "student_dorm_share"] = student_share
    exclude = (buildings["student_dorm_share"] >= DORM_EXCLUDE_MIN_SHARE) & ~buildings["is_mixed_use"]
    buildings["exclude_student_dormitory"] = exclude

    matches["building_overlap_share"] = matches["overlap_m2"] / matches["footprint_m2"]
    matches["building_any_dorm_share"] = matches["building_index"].map(any_share.clip(upper=1.0))
    matches["building_student_dorm_share"] = matches["building_index"].map(buildings["student_dorm_share"])
    matches["decision"] = np.where(
        matches["building_index"].map(buildings["exclude_student_dormitory"]),
        "excluded_student_dormitory",
        np.where(
            matches["building_any_dorm_share"] >= DORM_REPORT_MIN_SHARE,
            "uncertain_kept",
            "negligible_overlap_kept",
        ),
    )
    reasons = []
    for row in matches.itertuples(index=False):
        if row.decision == "excluded_student_dormitory":
            reasons.append("student-signalled OSM dormitory covers >= 90% of a purely residential ALKIS building")
        elif row.decision == "negligible_overlap_kept":
            reasons.append("overlap below 10% of the ALKIS footprint")
        elif row.is_mixed_use:
            reasons.append("ALKIS mixed-use building")
        elif row.osm_class != "student_dormitory":
            reasons.append(f"OSM object is {row.osm_class}")
        else:
            reasons.append("partial overlap below 90% of the ALKIS footprint")
    matches["reason"] = reasons
    return buildings, matches.drop(columns="building_index")


def apply_student_residence_decisions(
    buildings: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    path: str | Path = STUDENT_DECISIONS_FILE,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Replace the automatic dormitory rule with the reviewed per-building decisions.

    Every building with a non-negligible OSM dormitory overlap must have a
    reviewed row; only rows with ``decision == 'exclude'`` receive zero weight.
    """
    if (
        STUDENT_DECISIONS_SHA256
        and Path(path) == STUDENT_DECISIONS_FILE
        and sha256_file(path) != STUDENT_DECISIONS_SHA256
    ):
        raise RuntimeError(f"Checksum mismatch for {path}; re-record it after an intentional review.")
    decisions = pd.read_csv(path, dtype={"building_id": str})
    if decisions["building_id"].duplicated().any() or not decisions["decision"].isin(["exclude", "keep"]).all():
        raise ValueError("Decision file must have one exclude/keep row per building.")
    unknown = sorted(set(decisions["building_id"]) - set(buildings["building_id"]))
    if unknown:
        raise ValueError(f"Decisions refer to buildings that are not eligible ALKIS buildings: {unknown[:5]}")
    to_review = (
        set(matches.loc[matches["decision"] != "negligible_overlap_kept", "building_id"]) if len(matches) else set()
    )
    unreviewed = sorted(to_review - set(decisions["building_id"]))
    if unreviewed:
        raise ValueError(f"OSM dormitory matches without a reviewed decision: {unreviewed[:5]}")
    result = buildings.copy()
    excluded = set(decisions.loc[decisions["decision"].eq("exclude"), "building_id"])
    result["exclude_student_dormitory"] = result["building_id"].isin(excluded)
    lookup = decisions.set_index("building_id")
    result["student_residence_decision"] = (
        result["building_id"].map(lookup["decision"]).fillna("not_reviewed_no_dormitory_match")
    )
    result["student_residence_category"] = result["building_id"].map(lookup["category"])
    audit = {
        "reviewed_buildings": int(len(decisions)),
        "excluded": int(len(excluded)),
        "kept": int((decisions["decision"] == "keep").sum()),
        "by_category": {
            f"{d} | {c}": int(n) for (d, c), n in decisions.groupby(["decision", "category"]).size().items()
        },
        "decision_file_sha256": STUDENT_DECISIONS_SHA256,
    }
    return result, audit


def allocation_weights(buildings: pd.DataFrame, method: str) -> pd.Series:
    """Building weights of one allocation method (equal, footprint or floor area)."""
    if method not in ALLOCATION_METHODS:
        raise ValueError(f"Unknown allocation method {method!r}; expected one of {ALLOCATION_METHODS}.")
    eligible = ~buildings["exclude_student_dormitory"].astype(bool)
    if method == "footprint":
        weight = buildings["footprint_m2"].astype(float)
    elif method == "equal":
        weight = pd.Series(1.0, index=buildings.index)
    else:
        weight = buildings["footprint_m2"].astype(float) * buildings["storeys"].astype(float)
    if (weight < 0).any() or weight.isna().any():
        raise ValueError("Allocation weights must be complete and non-negative.")
    return weight.where(eligible, 0.0)


def allocate_children(
    buildings: pd.DataFrame,
    children: pd.DataFrame,
    method: str = PRIMARY_ALLOCATION,
    columns: tuple[str, ...] = ("age_0_2", "age_3_5"),
) -> pd.DataFrame:
    """Allocate district child totals to buildings: ``D_b = C_d * w_b / sum_d(w)``.

    Each age band is allocated with the same weights; ``children_0_5`` is their
    sum. The function raises if any district total is not reproduced.
    """
    weight = allocation_weights(buildings, method)
    district_sum = weight.groupby(buildings["district_id"]).transform("sum")
    totals = children.set_index("district_id")
    missing = sorted(set(totals.index) - set(buildings.loc[weight > 0, "district_id"]))
    if missing:
        raise ValueError(f"Districts without eligible building weight: {missing}")
    share = weight / district_sum
    result = pd.DataFrame(index=buildings.index)
    result["allocation_weight"] = weight
    for column in columns:
        result[column] = share * buildings["district_id"].map(totals[column]).astype(float)
    result["children_0_5"] = result[list(columns)].sum(axis=1)

    check = result.groupby(buildings["district_id"])[list(columns) + ["children_0_5"]].sum()
    expected = totals[list(columns) + ["children_0_5"]].astype(float)
    error = (check - expected.loc[check.index]).abs().to_numpy().max()
    if error > 1e-6:
        raise AssertionError(f"District conservation failed for {method}: max abs error {error}.")
    return result


def build_demand_nodes(
    graph: nx.MultiDiGraph,
    buildings: gpd.GeoDataFrame,
    allocation: pd.DataFrame,
    snapping: pd.DataFrame,
    districts: gpd.GeoDataFrame,
    district_groups: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Aggregate allocated building demand to (network node, district) demand locations.

    ``estimated_demand`` holds the allocated projected children aged 0-5, the
    column the solvers read. Locations with zero demand are dropped.
    """
    frame = pd.DataFrame(
        {
            "nearest_node": snapping["nearest_node"],
            DISTRICT_NAME: buildings[DISTRICT_NAME],
            "district_id": buildings["district_id"],
            "footprint_m2": buildings["footprint_m2"],
            "excluded_dormitory": buildings["exclude_student_dormitory"].astype(bool),
            "age_0_2": allocation["age_0_2"],
            "age_3_5": allocation["age_3_5"],
            "estimated_demand": allocation["children_0_5"],
        }
    )
    frame["eligible_building"] = ~frame["excluded_dormitory"]
    demand = frame.groupby(["nearest_node", DISTRICT_NAME], as_index=False).agg(
        district_id=("district_id", "first"),
        represented_buildings=("eligible_building", "sum"),
        excluded_dormitory_buildings=("excluded_dormitory", "sum"),
        footprint_m2=("footprint_m2", "sum"),
        age_0_2=("age_0_2", "sum"),
        age_3_5=("age_3_5", "sum"),
        estimated_demand=("estimated_demand", "sum"),
    )
    demand = demand.loc[demand["estimated_demand"] > 0].reset_index(drop=True)
    vulnerability = districts.set_index(DISTRICT_NAME)["childcare_vulnerability_index"]
    demand["childcare_vulnerability_index"] = demand[DISTRICT_NAME].map(vulnerability)
    demand["rawls_access_multiplier"] = 1.0 + demand["childcare_vulnerability_index"]
    demand["rawls_priority_weight"] = demand["rawls_access_multiplier"]
    # There is no resident estimate; the column is kept for the downstream
    # table builders and is intentionally empty.
    demand["estimated_residents"] = np.nan
    demand["demand_definition"] = "projected 2026 children aged 0-5, footprint-allocated"
    demand["demand_id"] = range(len(demand))
    demand["demand_label"] = [f"demand_{index}" for index in range(len(demand))]
    node_frame = pd.DataFrame(
        {
            "nearest_node": list(graph.nodes),
            "x": [graph.nodes[node]["x"] for node in graph.nodes],
            "y": [graph.nodes[node]["y"] for node in graph.nodes],
        }
    )
    demand = demand.merge(node_frame, on="nearest_node", how="left")
    demand = gpd.GeoDataFrame(demand, geometry=gpd.points_from_xy(demand["x"], demand["y"]), crs=WGS84)
    return attach_vulnerability_groups_to_demand(demand, district_groups, DISTRICT_NAME)


def load_census_children_cells(districts: gpd.GeoDataFrame) -> pd.DataFrame:
    """Census children aged 0-5 in grid cells whose centre lies in a district."""
    c, cell = CENSUS, CENSUS_CELL_M
    d3035 = districts.to_crs(CENSUS_CRS)
    minx, miny, maxx, maxy = d3035.total_bounds
    keep = []
    with zipfile.ZipFile(CENSUS_ZIP) as archive:
        name = next(n for n in archive.namelist() if n.endswith(c["member_suffix"]))
        wanted = {col.upper() for col in [c["id_column"], c["x_column"], c["y_column"], *c["count_columns"]]}
        with archive.open(name) as handle:
            reader = pd.read_csv(
                io.TextIOWrapper(handle, encoding=c["encoding"], errors="replace"),
                sep=c["separator"],
                dtype=str,
                chunksize=500_000,
                usecols=lambda column: column.upper() in wanted,
            )
            for chunk in reader:
                x = pd.to_numeric(chunk[c["x_column"]])
                y = pd.to_numeric(chunk[c["y_column"]])
                keep.append(chunk[(x >= minx - cell) & (x <= maxx + cell) & (y >= miny - cell) & (y <= maxy + cell)])
    cells = pd.concat(keep, ignore_index=True)
    for column in c["count_columns"]:
        # "–" marks exactly zero or perturbed to zero (Cell-Key Method).
        cells[column] = pd.to_numeric(cells[column].str.replace(r"[^\d-]", "", regex=True), errors="coerce").fillna(0.0)
    cells["children_0_5"] = cells[c["count_columns"]].sum(axis=1)
    cells["x"] = pd.to_numeric(cells[c["x_column"]])
    cells["y"] = pd.to_numeric(cells[c["y_column"]])
    cells["gx"] = ((cells["x"] - cell / 2) // cell).astype(int)
    cells["gy"] = ((cells["y"] - cell / 2) // cell).astype(int)
    points = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells["x"], cells["y"]), crs=CENSUS_CRS)
    points = gpd.sjoin(points, d3035[["district_id", "geometry"]], predicate="within", how="inner")
    return pd.DataFrame(points.drop(columns=["geometry", "index_right"]))


def building_cells(buildings: pd.DataFrame) -> pd.DataFrame:
    """Census grid cell (index in the census CRS) of every building's representative point."""
    pts = gpd.GeoSeries(gpd.points_from_xy(buildings["rep_x"], buildings["rep_y"]), crs=PROJECTED_CRS).to_crs(
        CENSUS_CRS
    )
    return pd.DataFrame(
        {
            "gx": (pts.x // CENSUS_CELL_M).astype(int).to_numpy(),
            "gy": (pts.y // CENSUS_CELL_M).astype(int).to_numpy(),
        },
        index=buildings.index,
    )


def census_hybrid_allocation(
    buildings: pd.DataFrame,
    children: pd.DataFrame,
    cells: pd.DataFrame,
    excluded_cells: frozenset[str] = frozenset(),
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Return (children per building, fallback cells, audit).

    ``excluded_cells`` lists census cell IDs whose children are removed from the within-district shares
    (a documented robustness assumption); the district totals are still fully allocated.
    """
    C = children.set_index("district_id")["children_0_5"]
    use = cells[~cells[CENSUS["id_column"]].isin(excluded_cells)]
    P = use.groupby(KEY)["children_0_5"].sum()
    P = P[P > 0]
    if not P.groupby(level=0).sum().reindex(C.index).gt(0).all():
        raise ValueError("A district has no positive census children; census-grid allocation is undefined there.")
    Dg = P / P.groupby(level=0).transform("sum") * C.reindex(P.index.get_level_values(0)).to_numpy()

    cell_of = building_cells(buildings)
    eligible = ~buildings["exclude_student_dormitory"].astype(bool)
    frame = pd.DataFrame({"district_id": buildings["district_id"], **cell_of, "A": buildings["footprint_m2"]})[eligible]
    in_cell = pd.MultiIndex.from_frame(frame[KEY]).isin(Dg.index)
    hit = frame[in_cell]
    share = hit["A"] / hit.groupby(KEY)["A"].transform("sum")
    D = pd.Series(0.0, index=buildings.index)
    D[hit.index] = share.to_numpy() * Dg.reindex(pd.MultiIndex.from_frame(hit[KEY])).to_numpy()

    fb = Dg.index.difference(pd.MultiIndex.from_frame(hit[KEY]).unique())
    centres = cells.set_index(KEY).loc[fb]
    fallback = pd.DataFrame(
        {
            "district_id": fb.get_level_values(0),
            "gx": fb.get_level_values(1),
            "gy": fb.get_level_values(2),
            "cell_id": centres[CENSUS["id_column"]].to_numpy(),
            "x_3035": centres["x"].to_numpy(),
            "y_3035": centres["y"].to_numpy(),
            "census_children": P[fb].to_numpy(),
            "children_0_5": Dg[fb].to_numpy(),
        }
    )
    audit = {
        "census_cells_with_children": int(len(P)),
        "excluded_cells": sorted(excluded_cells),
        "buildings_with_demand": int((D > 0).sum()),
        "eligible_buildings_without_demand": int(eligible.sum() - (D > 0).sum()),
        "fallback_cells": int(len(fallback)),
        "fallback_children": float(fallback["children_0_5"].sum()),
        "max_district_error": float(
            (
                D.groupby(buildings["district_id"])
                .sum()
                .add(fallback.groupby("district_id")["children_0_5"].sum(), fill_value=0)
                - C
            )
            .abs()
            .max()
        ),
    }
    return D, fallback, audit
