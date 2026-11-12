"""Q4 district-level Estimated-Demand-adjusted maximin model.

This module keeps the same demand locations, sparse candidate relations, existing Kitas, and nearest-open accessibility
rule, but protects a Q4 district rather than an individual demand location.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .all_demand_candidates import (
    SparseCandidateRelations,
    evaluate_selected_sparse_solution,
)
from .core import (
    DISTRICT_NAME,
    GUROBI_OUTPUT_FLAG,
    GUROBI_TIME_LIMIT_SECONDS,
    RAWLS_STAGE1_MIP_GAP,
    LocationModelResult,
    _gurobi_status_name,
    _has_usable_gurobi_solution,
    _require_gurobi,
    weighted_mean,
)


def q4_district_demand_adjusted_accessibility(
    assignment: pd.DataFrame,
    district_name_column: str = DISTRICT_NAME,
    q4_column: str = "least_advantaged",
    time_column: str = "walk_time_min",
) -> pd.DataFrame:
    """Calculate the district accessibility metric of the District Rawlsian model.

    For each Q4 district ``d`` this returns:

    ``A_d = sum(D_i * t_i) / sum(D_i)``.

    The demand locations remain individual observations.  Aggregation occurs
    only after their nearest-open walking times have been calculated.
    """
    required = {
        district_name_column,
        q4_column,
        time_column,
        "estimated_demand",
    }
    missing = sorted(required.difference(assignment.columns))
    if missing:
        raise KeyError(f"Missing columns for district accessibility: {missing}")

    q4 = assignment.loc[assignment[q4_column].astype(bool)].copy()
    if q4.empty:
        raise ValueError("No Q4 demand locations are available.")

    rows = []
    for district_name, group in q4.groupby(district_name_column, sort=True):
        demand_total = float(group["estimated_demand"].sum())
        if demand_total <= 0.0:
            raise ValueError(f"Q4 district {district_name!r} has no positive Estimated Demand.")
        rows.append(
            {
                district_name_column: district_name,
                "demand_locations": int(len(group)),
                "total_estimated_demand": demand_total,
                "ordinary_mean_walk_time": float(group[time_column].mean()),
                "estimated_demand_adjusted_mean_walk_time": weighted_mean(
                    group[time_column], group["estimated_demand"]
                ),
                "median_walk_time": float(group[time_column].median()),
                "p90_walk_time": float(group[time_column].quantile(0.90)),
                "maximum_walk_time": float(group[time_column].max()),
            }
        )
    return pd.DataFrame(rows)


def _independent_nearest_open_times(
    graph: nx.MultiDiGraph,
    demand: gpd.GeoDataFrame,
    current_access: pd.DataFrame,
    candidate_sites: gpd.GeoDataFrame,
    selected_candidate_ids: list[str],
) -> pd.Series:
    """Recompute nearest-open times independently with network Dijkstra."""
    reverse_graph = graph.reverse(copy=False) if graph.is_directed() else graph
    demand_nodes = demand[["demand_id", "nearest_node"]].copy()
    current = current_access.set_index("demand_id")["current_walk_time"].astype(float).copy()
    selected = candidate_sites.loc[candidate_sites["facility_id"].astype(str).isin(map(str, selected_candidate_ids))]
    for row in selected.itertuples(index=False):
        lengths = nx.single_source_dijkstra_path_length(
            reverse_graph,
            row.nearest_node,
            weight="walk_time_min",
        )
        candidate_times = demand_nodes["nearest_node"].map(lengths).fillna(np.inf).astype(float)
        candidate_times.index = demand_nodes["demand_id"].to_numpy()
        current = np.minimum(current, candidate_times)
    current.name = "independent_nearest_open_walk_time"
    return current


def solve_sparse_district_demand_maximin(
    demand: gpd.GeoDataFrame,
    current_access: pd.DataFrame,
    relations: SparseCandidateRelations,
    p_new_kitas: int = 3,
    district_name_column: str = DISTRICT_NAME,
    q4_column: str = "least_advantaged",
    time_limit_seconds: float | None = None,
    warm_start_candidate_nodes: Iterable[object] | None = None,
    mip_focus: int = 1,
    model_artifact_dir: str | Path | None = None,
    model_artifact_stem: str = "q4_district_maximin",
    mip_gap: float | None = None,
    threads: int | None = None,
    nodefile_start: float | None = None,
    nodefile_dir: str | Path | None = None,
) -> LocationModelResult:
    """Solve Q4 district-level demand-adjusted maximin.

    The MILP uses one nearest-open selector for each Q4 demand location.  For
    each district, the linear expression ``sum(D_i*t_i) / sum(D_i)`` is bounded
    by ``Z``.  Minimizing ``Z`` therefore minimizes the worst Q4 district's
    demand-adjusted mean walking time.
    """
    gp, GRB = _require_gurobi()
    if mip_focus not in {0, 1, 2, 3}:
        raise ValueError("mip_focus must be one of Gurobi's values 0, 1, 2, or 3.")
    start = time.perf_counter()
    demand_lookup = demand.set_index("demand_id")
    current_time = current_access.set_index("demand_id")["current_walk_time"].astype(float)
    q4 = demand.loc[demand[q4_column].astype(bool)].copy()
    q4_ids = [int(value) for value in q4["demand_id"]]
    if q4.empty:
        raise ValueError("No Q4 demand locations were found.")

    q4_by_district = {
        district_name: group["demand_id"].astype(int).tolist()
        for district_name, group in q4.groupby(district_name_column, sort=True)
    }
    district_demand_totals = {
        district_name: float(demand_lookup.loc[demand_ids, "estimated_demand"].astype(float).sum())
        for district_name, demand_ids in q4_by_district.items()
    }
    if any(value <= 0.0 for value in district_demand_totals.values()):
        raise ValueError("Every Q4 district must have positive Estimated Demand.")

    candidate_indices = list(range(len(relations.candidate_ids)))
    model = gp.Model("q4_district_estimated_demand_maximin")
    artifact_dir = Path(model_artifact_dir) if model_artifact_dir is not None else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    model.Params.OutputFlag = 1 if artifact_dir is not None else GUROBI_OUTPUT_FLAG
    effective_time_limit = GUROBI_TIME_LIMIT_SECONDS if time_limit_seconds is None else float(time_limit_seconds)
    if effective_time_limit <= 0.0:
        raise ValueError("time_limit_seconds must be positive.")
    model.Params.TimeLimit = effective_time_limit
    effective_mip_gap = RAWLS_STAGE1_MIP_GAP if mip_gap is None else float(mip_gap)
    if effective_mip_gap <= 0.0:
        raise ValueError("mip_gap must be positive.")
    model.Params.MIPGap = effective_mip_gap
    # MIPFocus=1 is the validated model setting.
    model.Params.MIPFocus = int(mip_focus)
    if threads is not None:
        if int(threads) < 1:
            raise ValueError("threads must be positive when provided.")
        model.Params.Threads = int(threads)
    if nodefile_start is not None:
        if float(nodefile_start) < 0.0:
            raise ValueError("nodefile_start must be non-negative when provided.")
        model.Params.NodefileStart = float(nodefile_start)
    if nodefile_dir is not None:
        Path(nodefile_dir).mkdir(parents=True, exist_ok=True)
        model.Params.NodefileDir = str(Path(nodefile_dir).resolve())

    open_candidate = model.addVars(
        candidate_indices,
        vtype=GRB.BINARY,
        name="open_candidate",
    )
    warm_start_indices: list[int] = []
    if warm_start_candidate_nodes is not None:
        warm_start_nodes = {str(node) for node in warm_start_candidate_nodes}
        warm_start_indices = [
            index for index, node in enumerate(relations.candidate_nodes) if str(node) in warm_start_nodes
        ]
        if len(warm_start_indices) != len(warm_start_nodes):
            raise ValueError(
                "The requested district-maximin warm-start nodes are not all "
                "present in the candidate pool. "
                f"Requested={sorted(warm_start_nodes)}, "
                f"matched={len(warm_start_indices)}."
            )
        if len(warm_start_indices) != int(p_new_kitas):
            raise ValueError(
                "The district-maximin warm start must contain exactly P new "
                f"facilities ({p_new_kitas}), got {len(warm_start_indices)}."
            )
        warm_start_index_set = set(warm_start_indices)
        for index in candidate_indices:
            open_candidate[index].Start = 1.0 if index in warm_start_index_set else 0.0
    measured_access = model.addVars(
        q4_ids,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="q4_nearest_open_walk_time",
    )
    worst_district_access = model.addVar(
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="max_q4_district_adjusted_mean",
    )
    model.addConstr(
        gp.quicksum(open_candidate[index] for index in candidate_indices) == p_new_kitas,
        name="choose_candidate_budget",
    )

    selector_count = 0
    relevant_counts: list[int] = []
    for demand_id in q4_ids:
        selector_vars = []
        prefix_open = gp.LinExpr(0.0)
        for rank, (candidate_index, walk_time) in enumerate(relations.pairs_by_demand[demand_id]):
            selector = model.addVar(
                vtype=GRB.BINARY,
                name=f"nearest_open_{demand_id}_{rank}",
            )
            open_expr = open_candidate[candidate_index]
            model.addConstr(
                selector <= open_expr,
                name=f"nearest_requires_open_{demand_id}_{rank}",
            )
            model.addConstr(
                selector >= open_expr - prefix_open,
                name=f"first_open_selected_{demand_id}_{rank}",
            )
            selector_vars.append((selector, float(walk_time)))
            prefix_open += open_expr

        existing_selector = model.addVar(
            vtype=GRB.BINARY,
            name=f"nearest_existing_{demand_id}",
        )
        model.addConstr(
            existing_selector >= 1.0 - prefix_open,
            name=f"existing_selected_if_no_better_candidate_{demand_id}",
        )
        selector_vars.append((existing_selector, float(current_time.at[demand_id])))
        selector_count += len(selector_vars)
        relevant_counts.append(len(selector_vars))
        model.addConstr(
            gp.quicksum(selector for selector, _ in selector_vars) == 1,
            name=f"one_nearest_open_{demand_id}",
        )
        model.addConstr(
            measured_access[demand_id] == gp.quicksum(walk_time * selector for selector, walk_time in selector_vars),
            name=f"q4_nearest_open_time_definition_{demand_id}",
        )

    for district_name, demand_ids in q4_by_district.items():
        denominator = district_demand_totals[district_name]
        district_mean = (
            gp.quicksum(
                float(demand_lookup.at[demand_id, "estimated_demand"]) * measured_access[demand_id]
                for demand_id in demand_ids
            )
            / denominator
        )
        model.addConstr(
            district_mean <= worst_district_access,
            name=f"q4_district_maximin_bound_{district_name}",
        )

    model.setObjective(worst_district_access, GRB.MINIMIZE)

    log_path = None
    if artifact_dir is not None:
        log_path = artifact_dir / f"{model_artifact_stem}.log"
        model.Params.LogFile = str(log_path)
    model.update()
    model_stats = {
        "num_variables": int(model.NumVars),
        "num_binary_variables": int(model.NumBinVars),
        "num_continuous_variables": int(model.NumVars - model.NumBinVars),
        "num_constraints": int(model.NumConstrs),
        "num_nonzeros": int(model.NumNZs),
        "num_general_constraints": int(model.NumGenConstrs),
        "num_quadratic_constraints": int(model.NumQConstrs),
        "num_sos_constraints": int(model.NumSOS),
    }

    gap_trace: list[dict[str, float]] = []

    def record_mip_progress(callback_model: Any, where: int) -> None:
        """Gurobi callback: record incumbent and bound over time."""
        if where != GRB.Callback.MIP:
            return
        incumbent = float(callback_model.cbGet(GRB.Callback.MIP_OBJBST))
        bound = float(callback_model.cbGet(GRB.Callback.MIP_OBJBND))
        if not np.isfinite(incumbent) or not np.isfinite(bound):
            return
        denominator = max(abs(incumbent), 1e-12)
        gap_trace.append(
            {
                "runtime_seconds": float(callback_model.cbGet(GRB.Callback.RUNTIME)),
                "obj_val": incumbent,
                "obj_bound": bound,
                "mip_gap": abs(incumbent - bound) / denominator,
                "node_count": float(callback_model.cbGet(GRB.Callback.MIP_NODCNT)),
            }
        )

    model.optimize(record_mip_progress)
    status = _gurobi_status_name(model.Status, GRB)
    if not _has_usable_gurobi_solution(model, GRB):
        raise RuntimeError(f"Q4 district maximin has no usable solution: {status}")

    selected_indices = [index for index in candidate_indices if open_candidate[index].X > 0.5]
    if len(selected_indices) != int(p_new_kitas):
        raise RuntimeError(
            "Q4 district maximin selected an invalid number of facilities: "
            f"{len(selected_indices)} instead of {p_new_kitas}."
        )
    selected_ids = [relations.candidate_ids[index] for index in selected_indices]
    modeled_times = {demand_id: float(measured_access[demand_id].X) for demand_id in q4_ids}
    assignment = evaluate_selected_sparse_solution(
        demand,
        current_access,
        relations,
        selected_indices,
        model_stage2_access=None,
    )
    assignment["model_district_maximin_walk_time"] = assignment["demand_id"].map(modeled_times)

    district_values = q4_district_demand_adjusted_accessibility(
        assignment,
        district_name_column=district_name_column,
        q4_column=q4_column,
    )

    artifact_paths: dict[str, str] = {}
    if artifact_dir is not None:
        model_path = artifact_dir / f"{model_artifact_stem}.mps"
        model.write(str(model_path))
        artifact_paths["model_mps"] = str(model_path)
        if gap_trace:
            trace_path = artifact_dir / f"{model_artifact_stem}_gap_trace.csv"
            pd.DataFrame(gap_trace).to_csv(trace_path, index=False)
            artifact_paths["gap_trace"] = str(trace_path)
        try:
            solution_path = artifact_dir / f"{model_artifact_stem}.sol"
            model.write(str(solution_path))
            artifact_paths["solution"] = str(solution_path)
        except Exception:
            # A model file is always useful even when Gurobi cannot write a
            # solution file for a non-final status.
            pass
        try:
            start_path = artifact_dir / f"{model_artifact_stem}.mst"
            model.write(str(start_path))
            artifact_paths["mip_start"] = str(start_path)
        except Exception:
            pass
        if log_path is not None:
            artifact_paths["solver_log"] = str(log_path)

    return LocationModelResult(
        name="q4_district_estimated_demand_maximin",
        status=status,
        objective_value=float(model.ObjVal),
        selected_candidate_ids=selected_ids,
        assignment=assignment,
        solver_details={
            "status": status,
            "mip_gap": _safe_mip_gap(model),
            "mip_gap_tolerance": effective_mip_gap,
            "runtime_seconds": time.perf_counter() - start,
            "gurobi_runtime_seconds": float(model.Runtime),
            "time_limit_seconds": effective_time_limit,
            "threads": int(threads) if threads is not None else None,
            "nodefile_start": float(nodefile_start) if nodefile_start is not None else None,
            "nodefile_dir": str(nodefile_dir) if nodefile_dir is not None else None,
            "mip_focus": int(mip_focus),
            "objective_value": float(model.ObjVal),
            "obj_bound": float(model.ObjBound),
            "node_count": float(model.NodeCount),
            "incumbent_objective": float(model.ObjVal),
            "best_bound": float(model.ObjBound),
            "warm_start_applied": bool(warm_start_indices),
            "warm_start_network_nodes": " | ".join(map(str, warm_start_candidate_nodes or [])),
            "candidate_count": len(relations.candidate_ids),
            "q4_district_count": len(q4_by_district),
            "q4_demand_location_count": len(q4_ids),
            "facility_budget": int(p_new_kitas),
            "facility_budget_constraint": f"sum(open_candidate) == {int(p_new_kitas)}",
            "opening_variable_count": len(candidate_indices),
            "opening_variable_type": "BINARY",
            "nearest_open_selector_count": selector_count,
            "nearest_open_relevant_facility_count_mean": float(np.mean(relevant_counts)),
            "nearest_open_relevant_facility_count_max": int(max(relevant_counts)),
            "accessibility_rule": "nearest_open_facility",
            "existing_facilities_permanently_available": True,
            "objective_definition": (
                "minimize the maximum Q4 district Estimated-Demand-adjusted mean nearest-open walking time"
            ),
            "estimated_demand_in_vulnerability_index": False,
            "modeled_worst_q4_district_adjusted_mean": float(
                district_values["estimated_demand_adjusted_mean_walk_time"].max()
            ),
            "selected_candidate_count": len(selected_ids),
            "sparse_pair_count": relations.pair_count,
            "gap_trace": gap_trace,
            "artifact_paths": artifact_paths,
            **model_stats,
        },
    )


def validate_district_demand_maximin_solution(
    result: LocationModelResult,
    graph: nx.MultiDiGraph,
    demand: gpd.GeoDataFrame,
    current_access: pd.DataFrame,
    candidate_sites: gpd.GeoDataFrame,
    p_new_kitas: int = 3,
    district_name_column: str = DISTRICT_NAME,
    q4_column: str = "least_advantaged",
    tolerance: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independently validate access vectors and the district maximin objective."""
    independent = _independent_nearest_open_times(
        graph,
        demand,
        current_access,
        candidate_sites,
        result.selected_candidate_ids,
    )
    assignment = result.assignment.set_index("demand_id")
    modeled = assignment["walk_time_min"].astype(float).reindex(independent.index)
    abs_diff = (modeled - independent).abs()

    independent_frame = demand[["demand_id", district_name_column, q4_column, "estimated_demand"]].copy()
    independent_frame["walk_time_min"] = independent.reindex(independent_frame["demand_id"]).to_numpy()
    independent_districts = q4_district_demand_adjusted_accessibility(
        independent_frame,
        district_name_column=district_name_column,
        q4_column=q4_column,
    )
    modeled_frame = demand[["demand_id", district_name_column, q4_column, "estimated_demand"]].copy()
    modeled_frame["walk_time_min"] = modeled.reindex(modeled_frame["demand_id"]).to_numpy()
    modeled_districts = q4_district_demand_adjusted_accessibility(
        modeled_frame,
        district_name_column=district_name_column,
        q4_column=q4_column,
    )
    district_compare = modeled_districts.merge(
        independent_districts,
        on=district_name_column,
        suffixes=("_modeled", "_independent"),
        validate="one_to_one",
    )
    district_objective = float(independent_districts["estimated_demand_adjusted_mean_walk_time"].max())
    objective_difference = abs(float(result.objective_value) - district_objective)
    selected_count = len(result.selected_candidate_ids)
    q4_mask = demand[q4_column].astype(bool).to_numpy()
    independent_by_id = independent.reindex(demand["demand_id"]).to_numpy()
    q4_independent_times = independent_by_id[q4_mask]
    q4_demand = demand.loc[q4_mask, "estimated_demand"].astype(float)
    citywide_independent_demand_adjusted_mean = weighted_mean(
        pd.Series(independent_by_id),
        demand["estimated_demand"].astype(float),
    )
    q4_independent_demand_adjusted_mean = weighted_mean(
        pd.Series(q4_independent_times),
        q4_demand,
    )
    validation_pass = bool(
        selected_count == int(p_new_kitas) and float(abs_diff.max()) <= tolerance and objective_difference <= tolerance
    )
    validation = pd.DataFrame(
        [
            {
                "scenario": result.name,
                "solver_status": result.status,
                "solver_mip_gap": result.solver_details.get("mip_gap"),
                "runtime_seconds": result.solver_details.get("runtime_seconds"),
                "selected_candidate_count": selected_count,
                "expected_candidate_count": int(p_new_kitas),
                "modeled_vs_independent_max_abs_diff": float(abs_diff.max()),
                "modeled_vs_independent_mean_abs_diff": float(abs_diff.mean()),
                "modeled_vs_independent_mismatch_count": int((abs_diff > tolerance).sum()),
                "solver_objective_z": float(result.objective_value),
                "independent_worst_q4_district_adjusted_mean": district_objective,
                "objective_abs_diff": objective_difference,
                "independent_q4_maximum_walk_time": float(np.max(q4_independent_times)),
                "independent_q4_estimated_demand_adjusted_mean": q4_independent_demand_adjusted_mean,
                "independent_citywide_estimated_demand_adjusted_mean": citywide_independent_demand_adjusted_mean,
                "stage1_validation_pass": validation_pass,
            }
        ]
    )
    district_compare["district_adjusted_mean_abs_diff"] = (
        district_compare["estimated_demand_adjusted_mean_walk_time_modeled"]
        - district_compare["estimated_demand_adjusted_mean_walk_time_independent"]
    ).abs()
    return validation, district_compare


def _safe_mip_gap(model: Any) -> float | None:
    """The model's MIP gap, or None when Gurobi cannot report one."""
    try:
        return float(model.MIPGap)
    except Exception:
        return None
