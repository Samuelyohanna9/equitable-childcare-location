"""Candidate sources on the synthetic city: residential_buildings, file, demand_nodes."""

from __future__ import annotations

import copy
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import synthetic_schema2 as sv
from make_city import X0, Y0
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon, box
from test_workflow_synthetic import _NoSites, cli

from childcare_access import candidate_sources as cs
from childcare_access import solutions as sol
from childcare_access.comparison import fingerprint_inputs
from childcare_access.config import PROJECT_ROOT
from childcare_access.script_support import read_pickle

GOLDEN = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "golden_synthetic_inputs.json").read_text(encoding="utf-8"))


def build(data: dict, root, name: str):
    run = cli(sv.write(data, root / f"{name}.toml"), "build", "--workspace", str(root / name))
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    return root / name / "inputs"


def bundle(inputs):
    return read_pickle(inputs / "inputs_bundle.pkl")


def relations_by_node(inputs) -> dict:
    """Relations with candidate IDs replaced by their network nodes (comparable across ID schemes)."""
    r = read_pickle(inputs / "sparse_relations_all_demand.pkl")
    node = dict(zip(r.candidate_ids, r.candidate_nodes, strict=True))
    index = {i: node[c] for i, c in enumerate(r.candidate_ids)}
    return {d: sorted((index[c], t) for c, t in pairs) for d, pairs in r.pairs_by_demand.items()}


def screened_nodes(inputs) -> set:
    b = bundle(inputs)
    return set(b["candidates"]["nearest_node"].iloc[sorted(map(int, b["screen_retained_indices"]))])


def sites_frame(rows, crs=25832, **columns) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"site": [str(i) for i in range(len(rows))], **columns}, geometry=rows, crs=crs)


def spec(path, **extra) -> dict:
    return {
        "source": "file",
        "file": str(path),
        "id_column": "site",
        "max_snap_m": 100,
        "shared_node": "error",
        **extra,
    }


def with_candidates(data: dict, candidates: dict) -> dict:
    data = copy.deepcopy(data)
    data["candidates"] = candidates
    return data


@pytest.fixture(scope="module")
def city(tmp_path_factory):
    root = tmp_path_factory.mktemp("candidates")
    data = sv.base(root)
    inputs = build(data, root, "residential")
    return {
        "root": root,
        "data": data,
        "inputs": inputs,
        "bundle": bundle(inputs),
        "districts": bundle(inputs)["districts"],
        "graph": read_pickle(inputs / "walk_graph_base.pkl"),
        "buildings": gpd.read_file(inputs / "buildings_alkis_allocated.gpkg"),
    }


def run_supplied(city, frame, name, **extra):
    path = city["root"] / f"{name}.gpkg"
    frame.to_file(path, driver="GPKG")
    return cs.supplied(spec(path, **extra), city["districts"], city["graph"])


# ------------------------------------------------------------------------------------------------ equivalence
def test_residential_buildings_reproduce_the_default_pool(city):
    fresh = fingerprint_inputs(city["inputs"])
    assert all(fresh["exact"][k] == GOLDEN["exact"][k] for k in ("candidates", "relations", "screen"))
    sites = pd.read_csv(city["inputs"] / "candidate_sites.csv")
    assert len(sites) == len(city["bundle"]["candidates"]) and set(sites["status"]) == {"candidate"}


def test_supplied_points_at_the_building_locations_give_the_same_model_inputs(city):
    root, b = city["root"], city["buildings"]
    points = sites_frame(list(gpd.points_from_xy(b["rep_x"], b["rep_y"])), site=b["building_id"].astype(str).tolist())
    points.to_file(root / "buildings_as_sites.gpkg", driver="GPKG")
    inputs = build(with_candidates(city["data"], spec(root / "buildings_as_sites.gpkg")), root, "supplied")
    fresh, base = fingerprint_inputs(inputs), fingerprint_inputs(city["inputs"])
    assert all(fresh["exact"][k] == base["exact"][k] for k in ("demand", "current_access", "facilities", "children"))
    assert set(bundle(inputs)["candidates"]["nearest_node"]) == set(city["bundle"]["candidates"]["nearest_node"])
    assert relations_by_node(inputs) == relations_by_node(city["inputs"])  # same pairs and times by node
    assert screened_nodes(inputs) == screened_nodes(city["inputs"])
    assert bundle(inputs)["candidates"]["facility_id"].str.startswith("site:").all()  # original site IDs kept


# ------------------------------------------------------------------------------------------------ supplied sites
def test_sites_on_one_node_follow_a_deterministic_policy(city):
    near = [
        Point(X0 + 103, Y0 + 98),
        Point(X0 + 97, Y0 + 104),
        Point(X0 + 101, Y0 + 101),
    ]  # all snap to node (1,1)
    others = [Point(X0 + 400, Y0 + 400), Point(X0 + 500, Y0 + 100)]
    frame = sites_frame(near + others, site=["b", "a", "c", "x", "y"])
    with pytest.raises(cs.CandidateError, match="same network node"):
        run_supplied(city, frame, "shared_error")
    first, sites = run_supplied(city, frame, "shared_keep", shared_node="keep_first")
    shuffled, _ = run_supplied(city, frame.iloc[[4, 2, 0, 3, 1]], "shared_shuffled", shared_node="keep_first")
    assert first["site_id"].tolist() == shuffled["site_id"].tolist() == ["a", "x", "y"]
    assert sites.set_index("site_id").loc[["b", "c"], "reason"].tolist() == ["same network node as site a"] * 2


def test_eligibility_comes_only_from_the_source(city):
    frame = sites_frame(
        [
            Point(X0 + 100, Y0 + 100),
            Point(X0 + 200, Y0 + 100),
            Point(X0 + 400, Y0 + 400),
            Point(X0 + 500, Y0 + 500),
        ],
        state=["vacant", "built", None, "vacant"],
    )
    chosen, sites = run_supplied(city, frame, "eligible", eligible_column="state", eligible_values=["vacant"])
    assert chosen["site_id"].tolist() == ["0", "3"]
    reasons = sites.set_index("site_id")["reason"]
    assert reasons["1"].startswith("ineligible: state = built") and reasons["2"].startswith("no value in state")
    unconfigured, _ = run_supplied(city, frame, "as_supplied")
    assert len(unconfigured) == 4 and set(unconfigured["candidate_source"]) == {
        "supplied site (eligibility as given by the source)"
    }


def test_districts_boundaries_and_snapping_distance(city):
    frame = sites_frame(
        [
            Point(X0 + 300, Y0 + 150),
            Point(X0 - 200, Y0 - 200),
            Point(X0 + 150, Y0 + 150),
            Point(X0 + 100, Y0 + 400),
        ]
    )
    with pytest.raises(cs.CandidateError, match="exactly on a district boundary: \\['0'\\]"):
        run_supplied(city, frame, "where_default", max_snap_m=60)  # default boundary_site = "error"
    chosen, sites = run_supplied(city, frame, "where", max_snap_m=60, boundary_site="lowest_id")
    s = sites.set_index("site_id")
    assert s.loc["0", "on_boundary"] and s.loc["0", "district_id"] == "3"  # explicit: lowest district ID
    assert s.loc["0", "boundary_assignment"] == "lowest district ID (boundary_site = 'lowest_id')"
    assert (s.drop(index="0")["boundary_assignment"] == "").all()
    assert s.loc["1", "reason"] == "outside every district"
    assert s.loc["2", "reason"] == "more than max_snap_m = 60 m from the walking network"  # 70.7 m to the nearest node
    assert chosen["site_id"].tolist() == ["0", "3"] and (chosen["snap_distance_m"] <= 60).all()


def test_sites_on_a_disconnected_component_are_excluded(city):
    graph = city["graph"].copy()
    to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)
    (lon1, lat1), (lon2, lat2) = (
        to_wgs.transform(X0 + 150, Y0 + 150),
        to_wgs.transform(X0 + 160, Y0 + 150),
    )
    graph.add_node(-1, x=lon1, y=lat1)
    graph.add_node(-2, x=lon2, y=lat2)
    graph.add_edge(-1, -2, walk_time_min=0.1, length=10.0)
    graph.add_edge(-2, -1, walk_time_min=0.1, length=10.0)
    path = city["root"] / "island.gpkg"
    sites_frame([Point(X0 + 150, Y0 + 150), Point(X0 + 400, Y0 + 400)]).to_file(path, driver="GPKG")
    chosen, sites = cs.supplied(spec(path), city["districts"], graph)
    assert chosen["site_id"].tolist() == ["1"]
    assert (
        sites.set_index("site_id").loc["0", "reason"]
        == "on a network component that does not connect to the main walking network"
    )


def test_geometry_and_identifier_errors(city):
    cases = [
        (sites_frame([Point(X0 + 100, Y0 + 100)], crs=None), "no CRS"),
        (
            sites_frame([Point(X0 + 100, Y0 + 100), Point(X0 + 200, Y0 + 200)], site=["a", "a"]),
            "missing or duplicate site IDs",
        ),
        (
            sites_frame([LineString([(X0, Y0), (X0 + 10, Y0)])]),
            "supported geometries are Point, Polygon, MultiPolygon",
        ),
        (
            sites_frame(
                [
                    Polygon(
                        [
                            (X0 + 50, Y0 + 50),
                            (X0 + 150, Y0 + 150),
                            (X0 + 150, Y0 + 50),
                            (X0 + 50, Y0 + 150),
                        ]
                    )
                ]
            ),
            "invalid geometries",
        ),
        (sites_frame([box(X0 + 50, Y0 + 50, X0 + 150, Y0 + 150)]), "set polygon_point"),
    ]
    for i, (frame, message) in enumerate(cases):
        with pytest.raises(cs.CandidateError, match=message):
            run_supplied(city, frame, f"bad{i}")
    u_shape = Polygon(
        [
            (X0 + 50, Y0 + 50),
            (X0 + 250, Y0 + 50),
            (X0 + 250, Y0 + 250),
            (X0 + 210, Y0 + 250),
            (X0 + 210, Y0 + 90),
            (X0 + 90, Y0 + 90),
            (X0 + 90, Y0 + 250),
            (X0 + 50, Y0 + 250),
        ]
    )
    with pytest.raises(cs.CandidateError, match="centroid' lies outside"):
        run_supplied(city, sites_frame([u_shape]), "u_centroid", polygon_point="centroid")
    chosen, sites = run_supplied(
        city,
        sites_frame([u_shape, box(X0 + 380, Y0 + 380, X0 + 420, Y0 + 420)]),
        "u_point",
        polygon_point="representative_point",
    )
    assert len(chosen) == 2 and all(
        u_shape.buffer(1e-6).contains(Point(x, y)) for x, y in sites.loc[[0], ["x", "y"]].to_numpy()
    )


def test_too_few_candidates_are_reported_before_the_relations(city):
    root = city["root"]
    sites_frame([Point(X0 + 100, Y0 + 100), Point(X0 + 400, Y0 + 400)]).to_file(root / "two_sites.gpkg", driver="GPKG")
    config = sv.write(with_candidates(city["data"], spec(root / "two_sites.gpkg")), root / "two_sites.toml")
    run = cli(config, "build", "--workspace", str(root / "two_sites"))
    assert (
        run.returncode != 0 and "only 2 distinct candidate network node(s)" in run.stderr and "for P = 3" in run.stderr
    )
    assert not (root / "two_sites" / "inputs" / "sparse_relations_all_demand.pkl").exists()


# ------------------------------------------------------------------------------------------------ demand_nodes
def test_demand_node_candidates_are_labelled_experimental(city):
    inputs = build(with_candidates(city["data"], {"source": "demand_nodes"}), city["root"], "demand_nodes")
    b = bundle(inputs)
    assert set(b["candidates"]["candidate_source"]) == {cs.EXPERIMENTAL}
    assert set(b["candidates"]["base_node"]) == set(b["demand"]["nearest_node"])
    report = json.loads((inputs / "validation_report.json").read_text(encoding="utf-8"))
    assert report["candidates"] == {
        "source": "demand_nodes",
        "count": len(b["candidates"]),
        "experimental": True,
        "eligibility": "not assessed (modelling assumption)",
    }


# ------------------------------------------------------------------------------------------------ solving (Gurobi)
@pytest.mark.gurobi
def test_every_candidate_source_solves_and_supplied_building_sites_match_the_default_pool(city):
    root = city["root"]
    if not (root / "supplied").exists():
        test_supplied_points_at_the_building_locations_give_the_same_model_inputs(city)
    if not (root / "demand_nodes").exists():
        build(with_candidates(city["data"], {"source": "demand_nodes"}), root, "demand_nodes")
    results = {}
    for name in ("residential", "supplied", "demand_nodes"):
        run = cli(
            root / f"{name}.toml",
            "solve",
            "--workspace",
            str(root / name),
            "--p",
            "3",
            "--run-name",
            "P3",
        )
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        results[name] = sol.load_reported_solutions(
            run_dir=root / name / "runs" / "P3", config=_NoSites(), require_all=False
        )
        ids = set(bundle(root / name / "inputs")["candidates"]["facility_id"])
        assert all(set(s.sites) <= ids for s in results[name].values())
    for model in ("utilitarian", "individual", "district"):
        a, b = results["residential"][model], results["supplied"][model]
        assert (
            np.isclose(a.objective_value, b.objective_value, rtol=0, atol=1e-9) and a.certification == b.certification
        ), model
