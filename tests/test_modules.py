"""Package modules on small synthetic inputs: solution certification, validation checks and evaluation."""

from __future__ import annotations

from typing import ClassVar

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from childcare_access import evaluation, solutions, validation


# ================================================================================================== solutions
def _row(status, gap, incumbent=1.0):
    return pd.Series({"status": status, "mip_gap": gap, "incumbent_objective": incumbent})


def test_certification_is_derived_from_the_records():
    assert solutions.certify([_row("Optimal", 0.0)]) == solutions.PROVEN
    assert solutions.certify([_row("Optimal", 2.8e-9)]) == solutions.PROVEN  # the gap of the Münster District run
    assert solutions.certify([_row("Optimal", 0.005)]) == solutions.WITHIN_GAP  # optimal only within a 1% request
    assert solutions.certify([_row("Optimal", 0.0), _row("TimeLimit", 7.35)]) == solutions.STAGE2_TIME_LIMITED
    assert solutions.certify([_row("Optimal", 0.0), _row("SolutionLimit", 0.4)]) == solutions.STAGE2_NOT_PROVEN
    assert solutions.certify([_row("Optimal", 0.0), _row("Optimal", 0.0)]) == solutions.PROVEN
    assert solutions.certify([_row("TimeLimit", 0.2)]) == solutions.FEASIBLE
    assert solutions.certify([_row("Infeasible", None, None)]) == solutions.NO_SOLUTION


# ================================================================================================== validation
def _toy_districts():
    return gpd.GeoDataFrame({"district_id": ["01", "02"]}, geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=25832)


def test_crs_and_schema_checks():
    validation.require_columns(pd.DataFrame({"a": [1]}), ["a"], "t")
    with pytest.raises(validation.ValidationError, match=r"missing column\(s\) \['b'\]"):
        validation.require_columns(pd.DataFrame({"a": [1]}), ["a", "b"], "t")
    validation.require_crs(_toy_districts(), "districts", expected=25832)
    with pytest.raises(validation.ValidationError, match="expected EPSG:3035"):
        validation.require_crs(_toy_districts(), "districts", expected=3035)
    with pytest.raises(validation.ValidationError, match="not a projected CRS in metres"):
        validation.require_crs(_toy_districts().to_crs(4326), "districts")
    with pytest.raises(validation.ValidationError, match="no CRS"):
        validation.require_crs(gpd.GeoDataFrame(geometry=[Point(0, 0)]), "points")


def test_demand_conservation_check():
    totals = pd.Series({"01": 10.0, "02": 5.0})
    good = pd.DataFrame({"district_id": ["01", "01", "02"], "estimated_demand": [4.0, 6.0, 5.0]})
    assert validation.check_demand_conservation(good, totals)["max_abs_error"] == 0
    with pytest.raises(validation.ValidationError, match=r"not conserved in 1 district.*district 02"):
        validation.check_demand_conservation(good.assign(estimated_demand=[4.0, 6.0, 4.0]), totals)
    with pytest.raises(validation.ValidationError, match="missing or negative"):
        validation.check_demand_conservation(good.assign(estimated_demand=[4.0, -1.0, 5.0]), totals)


def test_district_membership_check():
    points = gpd.GeoDataFrame({"district_id": ["01", "02"]}, geometry=[Point(5, 5), Point(15, 5)], crs=25832)
    assert validation.check_district_membership(points, _toy_districts())["points"] == 2
    with pytest.raises(validation.ValidationError, match="1 point\\(s\\) recorded in another district"):
        validation.check_district_membership(points.assign(district_id=["01", "01"]), _toy_districts())
    with pytest.raises(validation.ValidationError, match="1 point\\(s\\) outside every district"):
        validation.check_district_membership(points.assign(geometry=[Point(5, 5), Point(50, 5)]), _toy_districts())


def _toy_graph():
    g = nx.DiGraph()
    for u, v in [(1, 2), (2, 3), (3, 4)]:
        g.add_edge(u, v, walk_time_min=1.0)
        g.add_edge(v, u, walk_time_min=1.0)
    g.add_edge(5, 1, walk_time_min=2.0)  # node 5 can leave but nothing leads back to it
    return g


def test_network_connectivity_check():
    g = _toy_graph()
    assert validation.check_network_connectivity(g, [1, 2, 5], [4])["demand_nodes_reaching_a_facility"] == 3
    with pytest.raises(validation.ValidationError, match="2 demand node\\(s\\) cannot reach"):  # 1 and 2
        validation.check_network_connectivity(g, [1, 2], [5])
    with pytest.raises(validation.ValidationError, match="not in the routing graph"):
        validation.check_network_connectivity(g, [1, 99], [4])
    with pytest.raises(validation.ValidationError, match="strongly\\) connected components"):
        validation.check_network_connectivity(g, [1], [4], base_graph=g)
    bad = g.copy()
    bad[1][2]["walk_time_min"] = -1.0
    with pytest.raises(validation.ValidationError, match="negative"):
        validation.check_network_connectivity(bad, [1], [4])


def test_candidate_and_solution_checks():
    candidates = pd.DataFrame({"facility_id": ["c1", "c2", "c3"], "nearest_node": [1, 2, 3]})
    assert validation.check_candidate_sites(candidates, _toy_graph())["candidates"] == 3
    with pytest.raises(validation.ValidationError, match="duplicated"):
        validation.check_candidate_sites(candidates.assign(facility_id=["c1", "c1", "c3"]), _toy_graph())
    with pytest.raises(validation.ValidationError, match="not in the routing graph"):
        validation.check_candidate_sites(candidates.assign(nearest_node=[1, 2, 77]), _toy_graph())
    assert validation.check_solution(["c1", "c3"], 2, candidates)["sites"] == ["c1", "c3"]
    with pytest.raises(validation.ValidationError, match="expected 2 distinct known sites"):
        validation.check_solution(["c1", "c1"], 2, candidates)


def test_optimisation_feasibility_check():
    candidates = pd.DataFrame({"facility_id": ["c1", "c2"]})
    access = pd.DataFrame(
        {
            "demand_id": [1, 2],
            "current_walk_time": [5.0, 7.0],
            "estimated_demand": [1.0, 2.0],
            "least_advantaged": [True, False],
            "district_name": ["A", "B"],
        }
    )

    class Relations:
        candidate_ids: ClassVar[list[str]] = ["c1", "c2"]
        pairs_by_demand: ClassVar[dict[int, list[tuple[int, float]]]] = {1: [(0, 3.0)], 2: [(1, 6.5)]}

    assert validation.check_optimisation_feasibility(2, candidates, access, Relations)["relation_pairs"] == 2
    with pytest.raises(validation.ValidationError, match="P = 3 exceeds the 2 candidate sites"):
        validation.check_optimisation_feasibility(3, candidates, access)
    with pytest.raises(validation.ValidationError, match="P = 2 exceeds the 1 candidates kept by the screen"):
        validation.check_optimisation_feasibility(2, candidates, access, screened_count=1)
    with pytest.raises(validation.ValidationError, match="least-advantaged group has no demand locations"):
        validation.check_optimisation_feasibility(1, candidates, access.assign(least_advantaged=False))
    with pytest.raises(validation.ValidationError, match="without a finite current walking time"):
        validation.check_optimisation_feasibility(1, candidates, access.assign(current_walk_time=[np.inf, 7.0]))
    Relations.pairs_by_demand = {1: [(0, 6.0)]}  # 6.0 is not an improvement on 5.0
    with pytest.raises(validation.ValidationError, match="not improving"):
        validation.check_optimisation_feasibility(1, candidates, access, Relations)


def test_assign_nearest_nodes_without_optional_sklearn_or_scipy():
    from childcare_access.core import assign_nearest_nodes

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(10, x=7.60, y=51.96)
    graph.add_node(20, x=7.70, y=52.02)
    points = gpd.GeoDataFrame(
        {"name": ["near_10", "near_20"]},
        geometry=gpd.points_from_xy([7.601, 7.699], [51.961, 52.019]),
        crs="EPSG:4326",
    )
    assert assign_nearest_nodes(graph, points)["nearest_node"].tolist() == [10, 20]


# ================================================================================================== evaluation
def test_nearest_open_times_on_a_toy_graph():
    times = evaluation.nearest_open_times(_toy_graph(), pd.Series({"a": 1, "b": 5, "c": 4}), [4])
    assert times.to_dict() == {"a": 3.0, "b": 5.0, "c": 0.0}
