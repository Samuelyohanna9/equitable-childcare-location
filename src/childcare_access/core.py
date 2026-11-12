"""Shared foundations of the analysis.

- Constants and model parameters of the active city configuration (CRS, walking speed, facility budget,
  Q4 quantile, solver limits).
- ``LocationModelResult``: the result container stored in the saved solution pickles (its class path must
  not change).
- The four-indicator, equal-weight vulnerability index and the district-level Q4 rule (used through
  :mod:`childcare_access.vulnerability`).
- OSM, network, snapping and current-accessibility helpers.
"""

import os
import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from .config import active_config

# Metric CRS for all local geometry, from the active city configuration (Münster: EPSG:25832).
PROJECTED_CRS = active_config().get("crs.projected")

# Model parameters of the active configuration ([model_parameters]; see PARAMETER_DEFAULTS for the defaults).
# See childcare_access.config.
PARAMETERS = active_config().parameters
WGS84 = 4326

# Walking speed in metres per minute (default 83.33 m/min, about 5 km/h; a standard planning assumption).
WALKING_SPEED_M_PER_MIN = PARAMETERS["walking_speed_m_per_min"]

DISTRICT_NAME = "district_name"

SINGLE_PARENT_COL = "pct_children_single_parent"
MIGRATION_CHILDREN_COL = "pct_children_migration_background"
CHILD_WELFARE_COL = "pct_welfare_0_14"
WELFARE_ADULT_COL = "pct_welfare_15_64"
VULNERABILITY_COLUMNS = [
    SINGLE_PARENT_COL,
    MIGRATION_CHILDREN_COL,
    CHILD_WELFARE_COL,
    WELFARE_ADULT_COL,
]

# OSM building tags read to screen buildings for student residences (the buildings themselves come from the
# building footprint file).
RESIDENTIAL_BUILDING_TYPES = [
    "apartments",
    "residential",
    "house",
    "detached",
    "semidetached_house",
    "terrace",
    "dormitory",
    "bungalow",
    "farm",
    "houseboat",
    "static_caravan",
    "cabin",
]

# Number of new facilities (facility budget P).
P_NEW_KITAS = PARAMETERS["facility_budget"]
WALK_NETWORK_TYPE = "walk"

# Least-advantaged group (Q4): districts at or above this quantile of the district vulnerability index (default:
# 0.75). The indicators are measured per district, so the group is defined per district; the number of demand
# nodes in a district does not affect who counts as least advantaged.
LEAST_ADVANTAGED_QUANTILE = PARAMETERS["least_advantaged_quantile"]

# Gurobi can report a lexicographic second stage as infeasible if the stage-1 optimum
# is reimposed with an unrealistically strict floating-point tolerance.
# These tiny relaxations are only numerical safeguards; they are not meant to
# substantively change the Rawlsian fairness target.
RAWLS_STAGE2_TOLERANCES = list(PARAMETERS["rawls_stage2_tolerances"])

# Gurobi time limit and relative MIP gaps ([model_parameters]); statuses and gaps are always reported.
GUROBI_TIME_LIMIT_SECONDS = PARAMETERS["time_limit_seconds"]
GUROBI_MIP_GAP = PARAMETERS["utilitarian_mip_gap"]
RAWLS_STAGE1_MIP_GAP = PARAMETERS["rawls_stage1_mip_gap"]
RAWLS_STAGE2_MIP_GAP = PARAMETERS["rawls_stage2_mip_gap"]
GUROBI_OUTPUT_FLAG = 0

# OSMnx: use the local cache (the saved OSM data), long request timeout, no console logging.
ox.settings.use_cache = True
ox.settings.requests_timeout = 180
ox.settings.log_console = False


@dataclass
class LocationModelResult:
    """Container for optimization results.

    - Stores solver status, objective value, chosen candidates, and final assignments.
    """

    name: str
    status: str
    objective_value: float
    selected_candidate_ids: list[str]
    assignment: pd.DataFrame
    secondary_objective_value: float | None = None
    solver_details: dict[str, object] = field(default_factory=dict)


def load_district_data(data_file: str, required_columns: list[str]) -> gpd.GeoDataFrame:
    """Load the district polygons and check that the required columns are present.

    - Reads the GeoJSON with GeoPandas.
    - Verifies that all required columns exist.
    - Reprojects to the metric CRS used for local analysis.
    """
    gdf = gpd.read_file(data_file)
    missing_columns = [column for column in required_columns if column not in gdf.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    if gdf.crs is None:
        raise ValueError("District data has no CRS.")
    return gdf.to_crs(PROJECTED_CRS)


def add_childcare_vulnerability_index(
    gdf: gpd.GeoDataFrame,
    vulnerability_columns: list[str],
    index_column: str = "childcare_vulnerability_index",
) -> gpd.GeoDataFrame:
    """Build the composite vulnerability index from the district social indicators.

    - Converts each vulnerability column to numeric (missing or non-numeric values raise).
    - Min-max normalises every indicator to [0, 1], so indicators on different scales become comparable
      while their ordering is preserved; a constant indicator is set to 0.
    - The index is the unweighted mean of the normalised indicators (equal weights: no indicator is
      given priority without a theoretical reason).
    """
    gdf = gdf.copy()

    for column in vulnerability_columns:
        gdf[column] = pd.to_numeric(gdf[column], errors="coerce")
        if gdf[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing or non-numeric values.")

        # Range = max - min. If range is zero, every district has the same value,
        # so the normalized version carries no discriminatory information.
        span = gdf[column].max() - gdf[column].min()
        norm_column = f"{column}_norm"
        if span == 0:
            gdf[norm_column] = 0.0
        else:
            gdf[norm_column] = (gdf[column] - gdf[column].min()) / span

    norm_columns = [f"{column}_norm" for column in vulnerability_columns]
    gdf[index_column] = gdf[norm_columns].mean(axis=1)
    return gdf


def build_district_vulnerability_groups(
    districts: gpd.GeoDataFrame,
    district_name_column: str,
    vulnerability_column: str = "childcare_vulnerability_index",
    least_advantaged_quantile: float = LEAST_ADVANTAGED_QUANTILE,
) -> pd.DataFrame:
    """Define district-level vulnerability quartiles and the least-advantaged group (Q4).

    - Uses one vulnerability value per district.
    - Calculates the requested percentile cutoff from the unique district values.
    - Flags districts at or above the cutoff as least advantaged.

    Demand or population is not used to define the group; it is used only later to measure how much
    demand is affected.
    """
    if not 0.0 < least_advantaged_quantile < 1.0:
        raise ValueError("least_advantaged_quantile must be between 0 and 1.")
    missing_columns = [
        column for column in [district_name_column, vulnerability_column] if column not in districts.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing columns for vulnerability groups: {missing_columns}")

    district_scores = (
        districts[[district_name_column, vulnerability_column]]
        .dropna(subset=[district_name_column, vulnerability_column])
        .groupby(district_name_column, as_index=False)[vulnerability_column]
        .mean()
    )
    if district_scores.empty:
        raise ValueError("No district vulnerability scores are available.")

    if district_scores[vulnerability_column].lt(0).any() or district_scores[vulnerability_column].gt(1).any():
        raise ValueError(f"{vulnerability_column} must be scaled between 0 and 1.")

    q25 = float(district_scores[vulnerability_column].quantile(0.25))
    q50 = float(district_scores[vulnerability_column].quantile(0.50))
    q75 = float(district_scores[vulnerability_column].quantile(0.75))
    least_advantaged_threshold = float(district_scores[vulnerability_column].quantile(least_advantaged_quantile))

    def quartile_label(value: float) -> str:
        """Quartile label (Q1-Q4) of an index value."""
        if value >= q75:
            return "Q4 most vulnerable"
        if value >= q50:
            return "Q3 upper-middle vulnerability"
        if value >= q25:
            return "Q2 lower-middle vulnerability"
        return "Q1 least vulnerable"

    result = district_scores.copy()
    result["vulnerability_quartile"] = result[vulnerability_column].map(quartile_label)
    result["least_advantaged_threshold"] = least_advantaged_threshold
    result["least_advantaged_quantile"] = float(least_advantaged_quantile)
    result["least_advantaged_district"] = result[vulnerability_column] >= least_advantaged_threshold
    result["least_advantaged"] = result["least_advantaged_district"]
    return result.sort_values(vulnerability_column, ascending=False).reset_index(drop=True)


def attach_vulnerability_groups_to_districts(
    districts: gpd.GeoDataFrame,
    district_vulnerability_groups: pd.DataFrame,
    district_name_column: str,
) -> gpd.GeoDataFrame:
    """Attach vulnerability quartile and least-advantaged flags to district polygons."""
    group_columns = [
        "vulnerability_quartile",
        "least_advantaged_threshold",
        "least_advantaged_quantile",
        "least_advantaged_district",
        "least_advantaged",
    ]
    result = districts.drop(
        columns=[column for column in group_columns if column in districts.columns],
        errors="ignore",
    ).copy()
    return result.merge(
        district_vulnerability_groups[[district_name_column, *group_columns]],
        on=district_name_column,
        how="left",
    )


def attach_vulnerability_groups_to_demand(
    demand_points: gpd.GeoDataFrame,
    district_vulnerability_groups: pd.DataFrame,
    district_name_column: str,
) -> gpd.GeoDataFrame:
    """Transfer district-level least-advantaged flags to the demand nodes.

    - Every demand node inherits the quartile of its district.
    - Nodes in Q4 (above-threshold) districts get least_advantaged=True.

    Accessibility is computed at node level, but vulnerability is known only per district: no
    within-district (household-level) vulnerability is assumed.
    """
    group_columns = [
        "vulnerability_quartile",
        "least_advantaged_threshold",
        "least_advantaged_quantile",
        "least_advantaged_district",
        "least_advantaged",
    ]
    result = demand_points.drop(
        columns=[column for column in group_columns if column in demand_points.columns],
        errors="ignore",
    ).copy()
    result = result.merge(
        district_vulnerability_groups[[district_name_column, *group_columns]],
        on=district_name_column,
        how="left",
    )
    result["least_advantaged"] = result["least_advantaged"].fillna(False).astype(bool)
    result["least_advantaged_district"] = result["least_advantaged_district"].fillna(False).astype(bool)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=demand_points.crs)


def get_bbox_wgs84(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Create a latitude/longitude bounding box for OSM downloads.

    - Reprojects the districts to WGS84.
    - Returns west, south, east, north bounds.
    """
    west, south, east, north = gdf.to_crs(WGS84).total_bounds
    return float(west), float(south), float(east), float(north)


def download_kindergartens(bbox: tuple[float, float, float, float], tags: dict | None = None) -> gpd.GeoDataFrame:
    """Download existing kindergarten locations from OSM.

    - Queries OSM amenity=kindergarten (or the given ``tags``).
    - Converts geometries to centroid points.
    - Creates stable IDs and readable names.
    """
    pois = ox.features_from_bbox(bbox=bbox, tags=tags or {"amenity": "kindergarten"})
    if pois.empty:
        raise ValueError("No kindergarten locations were returned by OSM.")

    pois = pois.to_crs(PROJECTED_CRS)
    pois["geometry"] = pois.geometry.centroid
    pois = pois.to_crs(WGS84)

    result = pois[["geometry"]].copy()
    if "name" in pois.columns:
        result["site_name"] = pois["name"]
    else:
        result["site_name"] = None

    result["site_name"] = result["site_name"].fillna(
        pd.Series(
            [f"Existing Kita {index + 1}" for index in range(len(result))],
            index=result.index,
        )
    )
    result["facility_type"] = "existing"
    result["facility_id"] = [f"existing_{index}" for index in range(len(result))]
    return result.reset_index(drop=True)


def assign_nearest_nodes(
    graph: nx.MultiDiGraph,
    points: gpd.GeoDataFrame,
    node_column: str = "nearest_node",
) -> gpd.GeoDataFrame:
    """Snap point features to their nearest network nodes.

    - Projects graph nodes and points to the local metric CRS.
    - Finds the nearest graph node with NumPy in memory-safe chunks.
    """
    result = points.to_crs(WGS84).copy()
    graph_crs = graph.graph.get("crs", WGS84)
    node_records = [
        {"node_id": node_id, "geometry": Point(data["x"], data["y"])} for node_id, data in graph.nodes(data=True)
    ]
    nodes = gpd.GeoDataFrame(node_records, geometry="geometry", crs=graph_crs).to_crs(PROJECTED_CRS)
    projected_points = result.to_crs(PROJECTED_CRS)

    node_ids = nodes["node_id"].to_numpy()
    node_x = nodes.geometry.x.to_numpy(dtype=float)
    node_y = nodes.geometry.y.to_numpy(dtype=float)
    point_x = projected_points.geometry.x.to_numpy(dtype=float)
    point_y = projected_points.geometry.y.to_numpy(dtype=float)

    nearest = np.empty(len(projected_points), dtype=node_ids.dtype)
    point_chunk_size = 256
    node_chunk_size = 4096
    for start in range(0, len(projected_points), point_chunk_size):
        end = min(start + point_chunk_size, len(projected_points))
        best_dist = np.full(end - start, np.inf)
        best_node_index = np.zeros(end - start, dtype=int)

        px = point_x[start:end]
        py = point_y[start:end]
        for node_start in range(0, len(nodes), node_chunk_size):
            node_end = min(node_start + node_chunk_size, len(nodes))
            dist2 = (px[:, None] - node_x[node_start:node_end][None, :]) ** 2 + (
                py[:, None] - node_y[node_start:node_end][None, :]
            ) ** 2
            local_index = np.argmin(dist2, axis=1)
            local_dist = dist2[np.arange(end - start), local_index]
            improved = local_dist < best_dist
            best_dist[improved] = local_dist[improved]
            best_node_index[improved] = node_start + local_index[improved]

        nearest[start:end] = node_ids[best_node_index]

    result[node_column] = nearest
    return result


def compute_travel_time_matrix(
    graph: nx.MultiDiGraph,
    demand_points: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    weight: str = "walk_time_min",
) -> pd.DataFrame:
    """Compute network travel times from all facilities to all demand nodes.

    - For each facility node, run Dijkstra shortest path over the walk graph.
    - Store the resulting travel times in a matrix with demand_id rows and facility columns.
    """
    facility_ids = facilities["facility_id"].tolist()
    facility_nodes = facilities["nearest_node"].tolist()
    unique_demand_nodes = pd.Index(demand_points["nearest_node"].unique(), name="nearest_node")
    reverse_graph = graph.reverse(copy=False) if graph.is_directed() else graph

    print(
        f"Computing travel times from {len(facility_ids)} facilities "
        f"to {len(unique_demand_nodes)} unique demand nodes..."
    )

    columns = {}
    for index, (facility_id, facility_node) in enumerate(zip(facility_ids, facility_nodes), start=1):
        lengths = nx.single_source_dijkstra_path_length(reverse_graph, facility_node, weight=weight)
        columns[facility_id] = pd.Series(
            (float(lengths.get(node, np.inf)) for node in unique_demand_nodes),
            index=unique_demand_nodes,
        )
        if index % 50 == 0 or index == len(facility_ids):
            print(f"  processed {index}/{len(facility_ids)} facilities")

    node_matrix = pd.DataFrame(columns)
    demand_matrix = (
        demand_points[["demand_id", "nearest_node"]]
        .merge(
            node_matrix,
            left_on="nearest_node",
            right_index=True,
            how="left",
        )
        .set_index("demand_id")
    )

    return demand_matrix[facility_ids]


def compute_current_accessibility(
    demand_points: gpd.GeoDataFrame,
    existing_kitas: gpd.GeoDataFrame,
    district_name_column: str,
    weight_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the baseline accessibility before any new kindergarten is added.

    - Takes the minimum current travel time to any existing kindergarten.
    - Calculates weighted by estimated demand and Rawls-weighted burdens.
    """
    current_walk_time = weight_matrix.min(axis=1)
    nearest_facility = weight_matrix.idxmin(axis=1)

    base_columns = [
        "demand_id",
        "demand_label",
        district_name_column,
        "represented_buildings",
        "estimated_residents",
        "estimated_demand",
        "rawls_access_multiplier",
        "rawls_priority_weight",
        "childcare_vulnerability_index",
    ]
    optional_columns = [
        "vulnerability_quartile",
        "least_advantaged_threshold",
        "least_advantaged_quantile",
        "least_advantaged_district",
        "least_advantaged",
    ]
    current_access = demand_points[
        base_columns + [column for column in optional_columns if column in demand_points.columns]
    ].copy()
    current_access["nearest_existing_kita"] = nearest_facility.values
    current_access["current_walk_time"] = current_walk_time.values
    current_access["current_demand_burden"] = current_access["current_walk_time"] * current_access["estimated_demand"]
    current_access["current_rawls_burden"] = (
        current_access["current_walk_time"] * current_access["rawls_access_multiplier"]
    )

    current_access = current_access.merge(
        existing_kitas[["facility_id", "site_name"]].rename(
            columns={"facility_id": "nearest_existing_kita", "site_name": "nearest_existing_name"}
        ),
        on="nearest_existing_kita",
        how="left",
    )
    return current_access


def _require_gurobi() -> tuple[Any, Any]:
    """Import Gurobi only when an optimization model is solved."""
    academic_license = Path.home() / "gurobi.lic"
    if academic_license.exists():
        os.environ["GRB_LICENSE_FILE"] = str(academic_license)

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise ImportError(
            "Gurobi is required because the Individual Rawlsian Stage 2 optimizes "
            "a quadratic standard-deviation objective. Install it with "
            "`uv sync` or `uv add gurobipy`, then ensure a valid Gurobi academic "
            "or named-user license is available."
        ) from error
    return gp, GRB


def _gurobi_status_name(status_code: int, GRB: Any) -> str:
    """Readable name of a Gurobi status code (e.g. ``Optimal``, ``TimeLimit``)."""
    status_names = {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.INF_OR_UNBD: "InfeasibleOrUnbounded",
        GRB.UNBOUNDED: "Unbounded",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.SUBOPTIMAL: "Suboptimal",
        GRB.INTERRUPTED: "Interrupted",
        GRB.NUMERIC: "Numeric",
    }
    return status_names.get(status_code, f"GurobiStatus{status_code}")


def _has_usable_gurobi_solution(model: Any, GRB: Any) -> bool:
    """Return True when Gurobi produced a solution that can be inspected."""
    return model.Status == GRB.OPTIMAL or (
        model.Status in {GRB.TIME_LIMIT, GRB.SUBOPTIMAL} and getattr(model, "SolCount", 0) > 0
    )


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Compute a weighted average."""
    if len(series) == 0 or float(weights.sum()) == 0.0:
        return float("nan")
    return float(np.average(series, weights=weights))
