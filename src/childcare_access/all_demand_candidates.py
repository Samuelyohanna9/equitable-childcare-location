"""Sparse all-demand-node candidate models and their inputs.

The common inputs, the improving candidate pairs of every demand location, the Utilitarian and the two-stage
Individual Rawlsian solvers, and the independent Dijkstra validation of a solution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .core import (
    DISTRICT_NAME,
    GUROBI_MIP_GAP,
    GUROBI_OUTPUT_FLAG,
    GUROBI_TIME_LIMIT_SECONDS,
    P_NEW_KITAS,
    RAWLS_STAGE1_MIP_GAP,
    RAWLS_STAGE2_MIP_GAP,
    RAWLS_STAGE2_TOLERANCES,
    LocationModelResult,
    _gurobi_status_name,
    _has_usable_gurobi_solution,
    _require_gurobi,
)


@dataclass
class SparseCandidateRelations:
    """Sparse distances from all-demand-node candidates to demand nodes."""

    pairs_by_demand: dict[int, list[tuple[int, float]]]
    candidate_ids: list[str]
    candidate_nodes: list[object]
    candidate_lookup: gpd.GeoDataFrame
    pair_count: int
    runtime_seconds: float
    max_current_walk_time: float


def build_all_demand_candidate_sites(demand: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Use every final demand node as a potential new-facility candidate."""
    candidates = demand[["demand_id", DISTRICT_NAME, "nearest_node", "geometry"]].copy()
    candidates["facility_id"] = candidates["demand_id"].map(lambda value: f"all_demand_candidate_{int(value)}")
    candidates["facility_type"] = "candidate"
    candidates["candidate_source"] = "all_demand_node"
    candidates["site_name"] = (
        candidates[DISTRICT_NAME].astype(str) + " demand node " + candidates["demand_id"].astype(str)
    )
    return gpd.GeoDataFrame(candidates, geometry="geometry", crs=demand.crs).reset_index(drop=True)


def compute_sparse_candidate_relations(
    graph: nx.MultiDiGraph,
    demand: gpd.GeoDataFrame,
    current_access: pd.DataFrame,
    candidate_sites: gpd.GeoDataFrame,
    max_sparse_pairs: int | None = None,
) -> SparseCandidateRelations:
    """Find only candidate-demand pairs that can beat the nearest existing Kita."""
    start = time.perf_counter()
    reverse_graph = graph.reverse(copy=False) if graph.is_directed() else graph

    current_time = current_access.set_index("demand_id")["current_walk_time"].astype(float)
    max_current_time = float(current_time.max())
    demand_nodes = demand[["demand_id", "nearest_node"]].copy()

    demand_ids_by_node: dict[object, list[int]] = {}
    for row in demand_nodes.itertuples(index=False):
        demand_ids_by_node.setdefault(row.nearest_node, []).append(int(row.demand_id))

    candidate_lookup = candidate_sites.reset_index(drop=True).copy()
    candidate_ids = candidate_lookup["facility_id"].astype(str).tolist()
    candidate_nodes = candidate_lookup["nearest_node"].tolist()

    candidate_indices_by_node: dict[object, list[int]] = {}
    for candidate_index, node in enumerate(candidate_nodes):
        candidate_indices_by_node.setdefault(node, []).append(candidate_index)

    pairs_by_demand: dict[int, list[tuple[int, float]]] = {int(demand_id): [] for demand_id in demand["demand_id"]}
    pair_count = 0

    for processed, (candidate_node, candidate_indices) in enumerate(
        candidate_indices_by_node.items(),
        start=1,
    ):
        lengths = nx.single_source_dijkstra_path_length(
            reverse_graph,
            candidate_node,
            cutoff=max_current_time,
            weight="walk_time_min",
        )
        for demand_node, walk_time in lengths.items():
            demand_ids = demand_ids_by_node.get(demand_node)
            if not demand_ids:
                continue
            for demand_id in demand_ids:
                if float(walk_time) < float(current_time.at[demand_id]):
                    for candidate_index in candidate_indices:
                        pairs_by_demand[demand_id].append((candidate_index, float(walk_time)))
                        pair_count += 1
                        if max_sparse_pairs is not None and pair_count > max_sparse_pairs:
                            raise MemoryError(
                                f"Sparse relationship count exceeded the safety limit of {max_sparse_pairs:,} pairs."
                            )

        if processed % 500 == 0 or processed == len(candidate_indices_by_node):
            print(
                "Sparse candidate discovery: "
                f"{processed}/{len(candidate_indices_by_node)} unique candidate nodes, "
                f"{pair_count:,} improving pairs"
            )

    for demand_id, pairs in pairs_by_demand.items():
        pairs.sort(key=lambda pair: pair[1])

    return SparseCandidateRelations(
        pairs_by_demand=pairs_by_demand,
        candidate_ids=candidate_ids,
        candidate_nodes=candidate_nodes,
        candidate_lookup=candidate_lookup,
        pair_count=pair_count,
        runtime_seconds=time.perf_counter() - start,
        max_current_walk_time=max_current_time,
    )


def solve_sparse_utilitarian(
    demand: gpd.GeoDataFrame,
    current_access: pd.DataFrame,
    relations: SparseCandidateRelations,
    p_new_kitas: int = P_NEW_KITAS,
) -> LocationModelResult:
    """Solve the all-demand-candidates utilitarian model from sparse relations."""
    gp, GRB = _require_gurobi()
    start = time.perf_counter()
    model = gp.Model("all_demand_candidates_utilitarian")
    model.Params.OutputFlag = GUROBI_OUTPUT_FLAG
    model.Params.TimeLimit = GUROBI_TIME_LIMIT_SECONDS
    model.Params.MIPGap = GUROBI_MIP_GAP

    demand_lookup = demand.set_index("demand_id")
    current_time = current_access.set_index("demand_id")["current_walk_time"].astype(float)
    candidate_indices = list(range(len(relations.candidate_ids)))
    demand_ids = [int(value) for value in demand["demand_id"]]

    open_candidate = model.addVars(candidate_indices, vtype=GRB.BINARY, name="open_candidate")
    use_existing = model.addVars(demand_ids, vtype=GRB.BINARY, name="use_existing")

    pair_keys = [
        (demand_id, candidate_index)
        for demand_id, pairs in relations.pairs_by_demand.items()
        for candidate_index, _ in pairs
    ]
    pair_costs = {
        (demand_id, candidate_index): walk_time
        for demand_id, pairs in relations.pairs_by_demand.items()
        for candidate_index, walk_time in pairs
    }
    use_candidate = model.addVars(pair_keys, vtype=GRB.BINARY, name="use_candidate")

    model.addConstr(
        gp.quicksum(open_candidate[j] for j in candidate_indices) == p_new_kitas,
        name="choose_candidate_budget",
    )
    for demand_id in demand_ids:
        feasible_candidates = [candidate_index for candidate_index, _ in relations.pairs_by_demand[demand_id]]
        model.addConstr(
            use_existing[demand_id]
            + gp.quicksum(use_candidate[demand_id, candidate_index] for candidate_index in feasible_candidates)
            == 1,
            name=f"assign_one_{demand_id}",
        )
        for candidate_index in feasible_candidates:
            model.addConstr(
                use_candidate[demand_id, candidate_index] <= open_candidate[candidate_index],
                name=f"open_before_assign_{demand_id}_{candidate_index}",
            )

    model.setObjective(
        gp.quicksum(
            demand_lookup.at[demand_id, "estimated_demand"] * current_time.at[demand_id] * use_existing[demand_id]
            for demand_id in demand_ids
        )
        + gp.quicksum(
            demand_lookup.at[demand_id, "estimated_demand"]
            * pair_costs[demand_id, candidate_index]
            * use_candidate[demand_id, candidate_index]
            for demand_id, candidate_index in pair_keys
        ),
        GRB.MINIMIZE,
    )
    model.optimize()

    status = _gurobi_status_name(model.Status, GRB)
    if not _has_usable_gurobi_solution(model, GRB):
        raise RuntimeError(f"All-demand utilitarian model has no usable solution: {status}")

    selected_indices = [j for j in candidate_indices if open_candidate[j].X > 0.5]
    selected_ids = [relations.candidate_ids[j] for j in selected_indices]
    assignment = evaluate_selected_sparse_solution(
        demand,
        current_access,
        relations,
        selected_indices,
        model_stage2_access=None,
    )

    return LocationModelResult(
        name="all_demand_candidates_utilitarian",
        status=status,
        objective_value=float(model.ObjVal),
        selected_candidate_ids=selected_ids,
        assignment=assignment,
        secondary_objective_value=None,
        solver_details={
            "status": status,
            "mip_gap": _safe_mip_gap(model),
            "runtime_seconds": time.perf_counter() - start,
            "candidate_count": len(relations.candidate_ids),
            "assignment_variable_count": len(pair_keys) + len(demand_ids),
            "sparse_pair_count": relations.pair_count,
        },
    )


def apply_resource_parameters(
    model: Any,
    threads: int | None = None,
    nodefile_start: float | None = None,
    nodefile_dir: str | Path | None = None,
) -> None:
    """Set memory/thread parameters only; ``None`` leaves Gurobi's defaults unchanged."""
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


def solve_sparse_rawlsian(
    demand: gpd.GeoDataFrame,
    current_access: pd.DataFrame,
    relations: SparseCandidateRelations,
    p_new_kitas: int = P_NEW_KITAS,
    stage2_threads: int | None = None,
    stage2_nodefile_start: float | None = None,
    stage2_nodefile_dir: str | Path | None = None,
) -> LocationModelResult:
    """Solve the all-demand-candidates two-stage Rawlsian model."""
    gp, GRB = _require_gurobi()
    demand_lookup = demand.set_index("demand_id")
    current_time = current_access.set_index("demand_id")["current_walk_time"].astype(float)
    candidate_indices = list(range(len(relations.candidate_ids)))
    least_demand_ids = [int(value) for value in demand.loc[demand["least_advantaged"].astype(bool), "demand_id"]]
    if not least_demand_ids:
        raise ValueError("No least-advantaged demand nodes found.")

    stage1_start = time.perf_counter()
    stage1 = gp.Model("all_demand_candidates_rawlsian_stage1")
    stage1.Params.OutputFlag = GUROBI_OUTPUT_FLAG
    stage1.Params.TimeLimit = GUROBI_TIME_LIMIT_SECONDS
    stage1.Params.MIPGap = RAWLS_STAGE1_MIP_GAP
    stage1.Params.MIPFocus = 1

    open_stage1 = stage1.addVars(candidate_indices, vtype=GRB.BINARY, name="open_candidate")
    use_existing = stage1.addVars(least_demand_ids, vtype=GRB.BINARY, name="use_existing")
    stage1_pair_keys = [
        (demand_id, candidate_index)
        for demand_id in least_demand_ids
        for candidate_index, _ in relations.pairs_by_demand[demand_id]
    ]
    stage1_pair_costs = {
        (demand_id, candidate_index): walk_time
        for demand_id in least_demand_ids
        for candidate_index, walk_time in relations.pairs_by_demand[demand_id]
    }
    use_candidate = stage1.addVars(stage1_pair_keys, vtype=GRB.BINARY, name="use_candidate")
    worst_time = stage1.addVar(lb=0.0, name="max_q4_walk_time")

    stage1.addConstr(
        gp.quicksum(open_stage1[j] for j in candidate_indices) == p_new_kitas,
        name="choose_candidate_budget",
    )
    for demand_id in least_demand_ids:
        feasible_candidates = [candidate_index for candidate_index, _ in relations.pairs_by_demand[demand_id]]
        assigned_time = current_time.at[demand_id] * use_existing[demand_id] + gp.quicksum(
            stage1_pair_costs[demand_id, candidate_index] * use_candidate[demand_id, candidate_index]
            for candidate_index in feasible_candidates
        )
        stage1.addConstr(
            use_existing[demand_id]
            + gp.quicksum(use_candidate[demand_id, candidate_index] for candidate_index in feasible_candidates)
            == 1,
            name=f"assign_one_{demand_id}",
        )
        for candidate_index in feasible_candidates:
            stage1.addConstr(
                use_candidate[demand_id, candidate_index] <= open_stage1[candidate_index],
                name=f"open_before_assign_{demand_id}_{candidate_index}",
            )
        stage1.addConstr(assigned_time <= worst_time, name=f"q4_bound_{demand_id}")

    stage1.setObjective(worst_time, GRB.MINIMIZE)
    stage1.optimize()
    stage1_status = _gurobi_status_name(stage1.Status, GRB)
    if not _has_usable_gurobi_solution(stage1, GRB):
        raise RuntimeError(f"All-demand Rawlsian Stage 1 has no usable solution: {stage1_status}")
    stage1_runtime = time.perf_counter() - stage1_start
    z_star = float(worst_time.X)

    stage2 = None
    measured_access = None
    stage2_tolerance_used = None
    stage2_status = "NotSolved"
    stage2_start = time.perf_counter()
    selector_count = 0
    relevant_counts: list[int] = []

    for tolerance in RAWLS_STAGE2_TOLERANCES:
        stage2 = gp.Model("all_demand_candidates_rawlsian_stage2")
        stage2.Params.OutputFlag = GUROBI_OUTPUT_FLAG
        stage2.Params.TimeLimit = GUROBI_TIME_LIMIT_SECONDS
        stage2.Params.MIPGap = RAWLS_STAGE2_MIP_GAP
        stage2.Params.MIPFocus = 1
        # Resource settings only (defaults when None); the formulation is unchanged.
        apply_resource_parameters(stage2, stage2_threads, stage2_nodefile_start, stage2_nodefile_dir)
        open_stage2 = stage2.addVars(candidate_indices, vtype=GRB.BINARY, name="open_candidate")
        for j in candidate_indices:
            open_stage2[j].Start = open_stage1[j].X

        stage2.addConstr(
            gp.quicksum(open_stage2[j] for j in candidate_indices) == p_new_kitas,
            name="choose_candidate_budget",
        )
        measured_access = stage2.addVars(
            least_demand_ids,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="q4_nearest_open_walk_time",
        )
        mean_access = stage2.addVar(lb=0.0, name="mean_q4_walk_time")
        bound_limit = z_star + max(tolerance, abs(z_star) * 1e-8)
        selector_count = 0
        relevant_counts = []

        for demand_id in least_demand_ids:
            selector_vars = []
            prefix_open = gp.LinExpr(0.0)
            relevant_pairs = relations.pairs_by_demand[demand_id]
            for rank, (candidate_index, walk_time) in enumerate(relevant_pairs):
                nearest_selector = stage2.addVar(
                    vtype=GRB.BINARY,
                    name=f"nearest_open_{demand_id}_{rank}",
                )
                open_expr = open_stage2[candidate_index]
                stage2.addConstr(
                    nearest_selector <= open_expr,
                    name=f"nearest_requires_open_{demand_id}_{rank}",
                )
                stage2.addConstr(
                    nearest_selector >= open_expr - prefix_open,
                    name=f"first_open_selected_{demand_id}_{rank}",
                )
                selector_vars.append((nearest_selector, float(walk_time)))
                prefix_open += open_expr

            existing_selector = stage2.addVar(
                vtype=GRB.BINARY,
                name=f"nearest_existing_{demand_id}",
            )
            stage2.addConstr(
                existing_selector >= 1.0 - prefix_open,
                name=f"existing_selected_if_no_better_candidate_{demand_id}",
            )
            selector_vars.append((existing_selector, float(current_time.at[demand_id])))

            selector_count += len(selector_vars)
            relevant_counts.append(len(selector_vars))
            stage2.addConstr(
                gp.quicksum(selector for selector, _ in selector_vars) == 1,
                name=f"one_nearest_open_{demand_id}",
            )
            stage2.addConstr(
                measured_access[demand_id]
                == gp.quicksum(walk_time * selector for selector, walk_time in selector_vars),
                name=f"q4_nearest_open_walk_time_def_{demand_id}",
            )
            stage2.addConstr(
                measured_access[demand_id] <= bound_limit,
                name=f"preserve_stage1_bound_{demand_id}",
            )

        stage2.addConstr(
            len(least_demand_ids) * mean_access
            == gp.quicksum(measured_access[demand_id] for demand_id in least_demand_ids),
            name="mean_q4_walk_time_def",
        )
        stage2.setObjective(
            gp.quicksum(
                (measured_access[demand_id] - mean_access) * (measured_access[demand_id] - mean_access)
                for demand_id in least_demand_ids
            ),
            GRB.MINIMIZE,
        )
        stage2.optimize()
        stage2_status = _gurobi_status_name(stage2.Status, GRB)
        if _has_usable_gurobi_solution(stage2, GRB):
            stage2_tolerance_used = tolerance
            break

    if stage2 is None or measured_access is None or not _has_usable_gurobi_solution(stage2, GRB):
        raise RuntimeError(f"All-demand Rawlsian Stage 2 has no usable solution: {stage2_status}")

    modeled_stage2_access = {demand_id: float(measured_access[demand_id].X) for demand_id in least_demand_ids}
    selected_indices = [j for j in candidate_indices if open_stage2[j].X > 0.5]
    selected_ids = [relations.candidate_ids[j] for j in selected_indices]
    assignment = evaluate_selected_sparse_solution(
        demand,
        current_access,
        relations,
        selected_indices,
        model_stage2_access=modeled_stage2_access,
    )

    stage2_sse = float(max(stage2.ObjVal, 0.0))
    stage2_sd = float(np.sqrt(stage2_sse / len(least_demand_ids)))

    status = (
        "Optimal"
        if stage1.Status == GRB.OPTIMAL and stage2.Status == GRB.OPTIMAL
        else f"{stage1_status}/{stage2_status}"
    )
    return LocationModelResult(
        name="all_demand_candidates_rawlsian",
        status=status,
        objective_value=z_star,
        selected_candidate_ids=selected_ids,
        assignment=assignment,
        secondary_objective_value=stage2_sd,
        solver_details={
            "status": status,
            "stage1_status": stage1_status,
            "stage1_mip_gap": _safe_mip_gap(stage1),
            "stage1_runtime_seconds": stage1_runtime,
            "stage1_max_walk_time_least_advantaged": z_star,
            "stage2_status": stage2_status,
            "stage2_mip_gap": _safe_mip_gap(stage2),
            "stage2_runtime_seconds": time.perf_counter() - stage2_start,
            "stage2_tolerance_used": stage2_tolerance_used,
            "stage2_least_advantaged_walk_time_std": stage2_sd,
            "stage2_accessibility_rule": "nearest_open_facility",
            "stage2_nearest_open_selector_count": selector_count,
            "stage2_relevant_facility_count_mean": float(np.mean(relevant_counts)),
            "stage2_relevant_facility_count_max": int(max(relevant_counts)),
            "candidate_count": len(relations.candidate_ids),
            "stage1_assignment_variable_count": len(stage1_pair_keys) + len(least_demand_ids),
            "stage2_selector_variable_count": selector_count,
            "sparse_pair_count": relations.pair_count,
        },
    )


def evaluate_selected_sparse_solution(
    demand: gpd.GeoDataFrame,
    current_access: pd.DataFrame,
    relations: SparseCandidateRelations,
    selected_candidate_indices: list[int],
    model_stage2_access: dict[int, float] | None,
) -> pd.DataFrame:
    """Evaluate selected all-demand candidates by nearest-open walking time."""
    demand_lookup = demand.set_index("demand_id")
    current_lookup = current_access.set_index("demand_id")
    selected_set = set(selected_candidate_indices)
    candidate_names = relations.candidate_lookup["site_name"].astype(str).tolist()
    candidate_ids = relations.candidate_ids

    records = []
    for demand_id in demand["demand_id"].astype(int):
        best_time = float(current_lookup.at[demand_id, "current_walk_time"])
        best_facility_id = str(current_lookup.at[demand_id, "nearest_existing_kita"])
        best_site_name = str(current_lookup.at[demand_id, "nearest_existing_name"])
        best_facility_type = "existing"
        for candidate_index, walk_time in relations.pairs_by_demand[demand_id]:
            if candidate_index in selected_set and float(walk_time) < best_time:
                best_time = float(walk_time)
                best_facility_id = candidate_ids[candidate_index]
                best_site_name = candidate_names[candidate_index]
                best_facility_type = "candidate"

        record = {
            "demand_id": demand_id,
            "demand_label": demand_lookup.at[demand_id, "demand_label"],
            DISTRICT_NAME: demand_lookup.at[demand_id, DISTRICT_NAME],
            "assigned_facility_id": best_facility_id,
            "assigned_site_name": best_site_name,
            "assigned_facility_type": best_facility_type,
            "represented_buildings": demand_lookup.at[demand_id, "represented_buildings"],
            "estimated_residents": demand_lookup.at[demand_id, "estimated_residents"],
            "estimated_demand": demand_lookup.at[demand_id, "estimated_demand"],
            "rawls_access_multiplier": demand_lookup.at[demand_id, "rawls_access_multiplier"],
            "rawls_priority_weight": demand_lookup.at[demand_id, "rawls_priority_weight"],
            "walk_time_min": best_time,
            "demand_burden": best_time * demand_lookup.at[demand_id, "estimated_demand"],
            "rawls_burden": best_time * demand_lookup.at[demand_id, "rawls_access_multiplier"],
        }
        for column in [
            "vulnerability_quartile",
            "least_advantaged_threshold",
            "least_advantaged_quantile",
            "least_advantaged_district",
            "least_advantaged",
        ]:
            if column in demand_lookup.columns:
                record[column] = demand_lookup.at[demand_id, column]
        if model_stage2_access is not None:
            record["model_stage2_walk_time"] = model_stage2_access.get(demand_id, np.nan)
        records.append(record)

    return pd.DataFrame(records)


def validate_selected_solution_with_dijkstra(
    graph: nx.MultiDiGraph,
    demand: gpd.GeoDataFrame,
    current_access: pd.DataFrame,
    candidate_sites: gpd.GeoDataFrame,
    result: LocationModelResult,
) -> pd.DataFrame:
    """Independently recompute nearest-open access for the selected candidates only."""
    reverse_graph = graph.reverse(copy=False) if graph.is_directed() else graph
    demand_nodes = demand[["demand_id", "nearest_node"]].copy()
    current_time = current_access.set_index("demand_id")["current_walk_time"].astype(float)
    selected = candidate_sites.loc[candidate_sites["facility_id"].isin(result.selected_candidate_ids)].copy()

    recomputed = current_time.copy()
    for row in selected.itertuples(index=False):
        lengths = nx.single_source_dijkstra_path_length(
            reverse_graph,
            row.nearest_node,
            weight="walk_time_min",
        )
        selected_times = demand_nodes["nearest_node"].map(lengths).fillna(np.inf).astype(float)
        recomputed = np.minimum(recomputed, pd.Series(selected_times.values, index=demand_nodes["demand_id"]))

    assignment = result.assignment.set_index("demand_id")
    abs_diff = (assignment["walk_time_min"] - recomputed).abs()
    rows = [
        {
            "scenario": result.name,
            "mismatch_count": int((abs_diff > 1e-4).sum()),
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
        }
    ]
    if "least_advantaged" in assignment.columns:
        q4 = assignment["least_advantaged"].astype(bool)
        rows[0]["evaluated_q4_max"] = float(assignment.loc[q4, "walk_time_min"].max())
        rows[0]["recomputed_q4_max"] = float(recomputed.loc[q4].max())
        rows[0]["evaluated_q4_sd"] = float(assignment.loc[q4, "walk_time_min"].std(ddof=0))
        rows[0]["recomputed_q4_sd"] = float(recomputed.loc[q4].std(ddof=0))
        if "model_stage2_walk_time" in assignment.columns:
            modeled = assignment.loc[q4, "model_stage2_walk_time"].dropna()
            rows[0]["modeled_stage2_q4_sd"] = float(modeled.std(ddof=0))
    return pd.DataFrame(rows)


def _safe_mip_gap(model: Any) -> float | None:
    """The model's MIP gap, or None when Gurobi cannot report one."""
    try:
        return float(model.MIPGap)
    except Exception:
        return None
