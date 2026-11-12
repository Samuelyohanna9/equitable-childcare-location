"""Optimisation inputs and models: candidate sites, sparse relations, the Q4 screen and model sizes.

Also the three solvers (Utilitarian, Individual Rawlsian two-stage, District Rawlsian). The solver functions are
defined in :mod:`childcare_access.all_demand_candidates` and :mod:`childcare_access.district_demand_maximin`
(their result classes are stored in the saved solution pickles) and re-exported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .all_demand_candidates import (  # noqa: F401  (re-exported)
    SparseCandidateRelations,
    apply_resource_parameters,
    build_all_demand_candidate_sites,
    compute_sparse_candidate_relations,
    solve_sparse_rawlsian,
    solve_sparse_utilitarian,
)
from .core import DISTRICT_NAME, LocationModelResult  # noqa: F401  (LocationModelResult re-exported)
from .district_demand_maximin import (  # noqa: F401  (re-exported)
    q4_district_demand_adjusted_accessibility,
    solve_sparse_district_demand_maximin,
    validate_district_demand_maximin_solution,
)
from .settings import NODEFILE_MIN_FREE_GB, SOLVER_RESOURCES


def build_candidate_pool(demand: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """All-demand-node candidate pool, one candidate per network node."""
    from .all_demand_candidates import build_all_demand_candidate_sites

    candidates = build_all_demand_candidate_sites(demand)
    candidates = candidates.drop_duplicates(subset=["nearest_node"]).reset_index(drop=True)
    candidates["union_origin"] = "all_demand_node"
    return candidates


def nodefile_preflight(min_free_gb: float = NODEFILE_MIN_FREE_GB) -> dict:
    """Create the node-file directories and confirm they are writable with enough free space."""
    import shutil

    report = {}
    for model, settings in SOLVER_RESOURCES.items():
        directory = settings.get("nodefile_dir")
        if not directory:
            continue
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_bytes(b"ok")
        probe.unlink()
        free_gb = shutil.disk_usage(path).free / 1e9
        if free_gb < min_free_gb:
            raise RuntimeError(f"Only {free_gb:.1f} GB free for node files at {path}.")
        report[model] = {
            "nodefile_dir": str(path.resolve()),
            "free_gb": round(free_gb, 1),
            "threads": settings["threads"],
            "nodefile_start": settings["nodefile_start"],
        }
    return report


@dataclass
class DominanceScreen:
    """Result of the Q4 dominance screen: per-candidate report, retained indices, dominator mapping, summary."""

    report: pd.DataFrame
    retained_indices: list[int]
    mapping: pd.DataFrame
    summary: dict


def _pointwise_dominance(profiles: np.ndarray, supports: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Weak and strict pointwise dominators of each candidate."""
    candidate_count, q4_count = profiles.shape
    inverted: list[set[int]] = [set() for _ in range(q4_count)]
    for candidate_index, support in enumerate(supports):
        for position in support:
            inverted[int(position)].add(candidate_index)
    weak = np.full(candidate_count, -1, dtype=np.int64)
    strict = np.full(candidate_count, -1, dtype=np.int64)
    everyone = set(range(candidate_count))
    for q_index, support_array in enumerate(supports):
        support = [int(value) for value in support_array]
        if not support:
            possible = everyone.copy()
        else:
            rarest = min(support, key=lambda position: len(inverted[position]))
            possible = inverted[rarest].copy()
            for position in support:
                possible.intersection_update(inverted[position])
                if not possible:
                    break
        possible.discard(q_index)
        for p_index in possible:
            if not np.all(profiles[p_index, support_array] >= profiles[q_index, support_array]):
                continue
            weak[q_index] = p_index
            if np.any(profiles[p_index] > profiles[q_index]):
                strict[q_index] = p_index
                break
        if strict[q_index] < 0 and weak[q_index] >= 0:
            p_index = int(weak[q_index])
            if np.any(profiles[p_index] > profiles[q_index]):
                strict[q_index] = p_index
    return weak, strict


def q4_dominance_screen(
    relations: SparseCandidateRelations, demand: pd.DataFrame, current_access: pd.DataFrame, p_new: int = 3
) -> DominanceScreen:
    """Strict pointwise Q4 improvement-vector dominance over the full candidate pool.

    The rule depends only on Q4 membership, current walking times and candidate
    walking times. It does not use demand weights, so it is valid for every
    allocation variant that shares the same demand-location set.
    """
    q4_ids = demand.loc[demand["least_advantaged"].astype(bool), "demand_id"].astype(int).tolist()
    position = {demand_id: index for index, demand_id in enumerate(q4_ids)}
    current = current_access.set_index("demand_id")["current_walk_time"].astype(float)
    q4_current = current.loc[q4_ids].to_numpy()
    candidate_count = len(relations.candidate_ids)
    profiles = np.zeros((candidate_count, len(q4_ids)), dtype=np.float64)
    for demand_id in q4_ids:
        column = position[demand_id]
        for candidate_index, walk_time in relations.pairs_by_demand[demand_id]:
            profiles[candidate_index, column] = q4_current[column] - float(walk_time)
    supports = [np.flatnonzero(row > 0.0) for row in profiles]
    weak, strict = _pointwise_dominance(profiles, supports)
    retained = [index for index in range(candidate_count) if strict[index] < 0]
    if len(retained) < p_new + 1:
        raise RuntimeError("Too few retained candidates for the exact budget argument.")

    retained_set = set(retained)
    lookup = relations.candidate_lookup.reset_index(drop=True)
    nodes = list(relations.candidate_nodes)
    mapping_rows = []
    for index in range(candidate_count):
        if index in retained_set:
            continue
        chain = [index]
        current_index = index
        while current_index not in retained_set:
            next_index = int(strict[current_index])
            if next_index < 0 or next_index in chain:
                raise RuntimeError(f"Dominance chain does not terminate for candidate {index}: {chain}")
            chain.append(next_index)
            current_index = next_index
        if str(nodes[index]) == str(nodes[current_index]):
            raise RuntimeError(f"Dominance replacement kept the same network site for candidate {index}.")
        mapping_rows.append(
            {
                "removed_facility_id": relations.candidate_ids[index],
                "retained_dominator_facility_id": relations.candidate_ids[current_index],
                "dominance_chain_length": len(chain) - 1,
                "removed_network_node": nodes[index],
                "retained_network_node": nodes[current_index],
            }
        )
    report = lookup[["facility_id", "demand_id", DISTRICT_NAME, "nearest_node"]].copy()
    report["q4_improvement_count"] = [len(support) for support in supports]
    report["no_q4_improvement"] = report["q4_improvement_count"].eq(0)
    report["strict_dominator_facility_id"] = [
        relations.candidate_ids[int(value)] if value >= 0 else pd.NA for value in strict
    ]
    report["retained"] = strict < 0
    summary = {
        "candidate_pool": int(candidate_count),
        "q4_demand_locations": int(len(q4_ids)),
        "q4_districts": int(demand.loc[demand["least_advantaged"].astype(bool), DISTRICT_NAME].nunique()),
        "no_q4_improvement_candidates": int(report["no_q4_improvement"].sum()),
        "weak_pointwise_dominated_candidates": int((weak >= 0).sum()),
        "strict_pointwise_dominated_candidates": int((strict >= 0).sum()),
        "retained_candidates": int(len(retained)),
        "mapping_chains_validated": int(len(mapping_rows)),
        "max_chain_length": int(max((row["dominance_chain_length"] for row in mapping_rows), default=0)),
    }
    return DominanceScreen(
        report=report,
        retained_indices=retained,
        mapping=pd.DataFrame(mapping_rows),
        summary=summary,
    )


def model_size_estimates(
    relations: SparseCandidateRelations,
    demand: pd.DataFrame,
    screened_indices: list[int] | None,
    current_access: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Exact row/column/nonzero counts implied by the existing solver formulations.

    The counts mirror ``solve_sparse_utilitarian``, ``solve_sparse_rawlsian``
    (Stage 1 and Stage 2) and ``solve_sparse_district_demand_maximin`` line by
    line, so they can be checked against a Gurobi log without building a model.
    """
    candidate_count = len(relations.candidate_ids)
    demand_ids = demand["demand_id"].astype(int).tolist()
    q4 = demand.loc[demand["least_advantaged"].astype(bool)]
    q4_ids = q4["demand_id"].astype(int).tolist()
    k_all = np.array([len(relations.pairs_by_demand[d]) for d in demand_ids], dtype=np.int64)
    k_q4 = np.array([len(relations.pairs_by_demand[d]) for d in q4_ids], dtype=np.int64)
    # Gurobi drops zero coefficients: a candidate at the demand location's own
    # node has walking time 0, and a location at an existing Kita node has
    # current time 0. Each removes one nonzero from the rows whose coefficients
    # are walking times (Rawlsian bound and definition rows).
    current = (
        current_access.set_index("demand_id")["current_walk_time"].astype(float)
        if current_access is not None
        else pd.Series(1.0, index=demand_ids)
    )
    zero_pairs_q4 = sum(1 for d in q4_ids for _, t in relations.pairs_by_demand[d] if float(t) == 0.0)
    zero_current_q4 = int((current.reindex(q4_ids) == 0.0).sum())
    rows = []
    pairs = int(k_all.sum())
    rows.append(
        {
            "model": "Utilitarian",
            "candidates": candidate_count,
            "demand_locations": len(demand_ids),
            "pairs": pairs,
            "columns": candidate_count + len(demand_ids) + pairs,
            "rows": 1 + len(demand_ids) + pairs,
            "nonzeros": candidate_count + len(demand_ids) + 3 * pairs,
        }
    )
    q_pairs = int(k_q4.sum())
    rows.append(
        {
            "model": "Individual Rawlsian Stage 1",
            "candidates": candidate_count,
            "demand_locations": len(q4_ids),
            "pairs": q_pairs,
            "columns": candidate_count + len(q4_ids) + q_pairs + 1,
            "rows": 1 + 2 * len(q4_ids) + q_pairs,
            "nonzeros": candidate_count + 3 * len(q4_ids) + 4 * q_pairs - zero_pairs_q4 - zero_current_q4,
        }
    )
    prefix = int((k_q4 * (k_q4 - 1) // 2).sum())
    rows.append(
        {
            "model": "Individual Rawlsian Stage 2",
            "candidates": candidate_count,
            "demand_locations": len(q4_ids),
            "pairs": q_pairs,
            "columns": candidate_count + len(q4_ids) + 1 + q_pairs + len(q4_ids),
            "rows": 2 + 2 * q_pairs + 4 * len(q4_ids),
            "nonzeros": candidate_count
            + (len(q4_ids) + 1)
            + 7 * q_pairs
            + prefix
            + 5 * len(q4_ids)
            - zero_pairs_q4
            - zero_current_q4,
        }
    )
    if screened_indices is not None:
        keep = set(screened_indices)
        k_s = np.array(
            [sum(1 for j, _ in relations.pairs_by_demand[d] if j in keep) for d in q4_ids],
            dtype=np.int64,
        )
        zero_pairs_s = sum(1 for d in q4_ids for j, t in relations.pairs_by_demand[d] if j in keep and float(t) == 0.0)
        s_pairs = int(k_s.sum())
        district_sizes = q4.groupby(DISTRICT_NAME).size()
        rows.append(
            {
                "model": "District Rawlsian (screened)",
                "candidates": len(keep),
                "demand_locations": len(q4_ids),
                "pairs": s_pairs,
                "columns": len(keep) + s_pairs + 2 * len(q4_ids) + 1,
                "rows": 1 + 2 * s_pairs + 3 * len(q4_ids) + len(district_sizes),
                "nonzeros": len(keep)
                + 7 * s_pairs
                + int((k_s * (k_s - 1) // 2).sum())
                + 4 * len(q4_ids)
                + int((district_sizes + 1).sum())
                - zero_pairs_s
                - zero_current_q4,
            }
        )
    return pd.DataFrame(rows)


def screened_relations(relations: SparseCandidateRelations, retained_indices: list[int]) -> SparseCandidateRelations:
    """Restrict relations to retained candidates, renumbering candidate indices."""
    from .all_demand_candidates import SparseCandidateRelations

    old_to_new = {old: new for new, old in enumerate(retained_indices)}
    pairs_by_demand = {
        demand_id: [(old_to_new[j], t) for j, t in pairs if j in old_to_new]
        for demand_id, pairs in relations.pairs_by_demand.items()
    }
    return SparseCandidateRelations(
        pairs_by_demand=pairs_by_demand,
        candidate_ids=[relations.candidate_ids[i] for i in retained_indices],
        candidate_nodes=[relations.candidate_nodes[i] for i in retained_indices],
        candidate_lookup=relations.candidate_lookup.iloc[retained_indices].reset_index(drop=True),
        pair_count=sum(len(p) for p in pairs_by_demand.values()),
        runtime_seconds=0.0,
        max_current_walk_time=relations.max_current_walk_time,
    )
