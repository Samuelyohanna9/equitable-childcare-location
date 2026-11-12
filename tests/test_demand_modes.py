"""Demand modes on the synthetic city: census_hybrid, building_area and direct (points, polygons).

Every version-2 build here uses the same generated raw data, so builds can be compared component by component.
"""

from __future__ import annotations

import copy
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import synthetic_schema2 as sv
from make_city import CHILDREN, X0, Y0
from shapely.geometry import MultiPoint, Point, box
from test_workflow_synthetic import cli

from childcare_access import demand_modes as dm
from childcare_access.comparison import fingerprint_inputs
from childcare_access.config import PROJECT_ROOT
from childcare_access.script_support import read_pickle

GOLDEN = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "golden_synthetic_inputs.json").read_text(encoding="utf-8"))
SOLVER_INPUTS = [k for k in GOLDEN["exact"] if k != "labels"]


def build(data: dict, root, name: str):
    config = sv.write(data, root / f"{name}.toml")
    run = cli(config, "build", "--workspace", str(root / name))
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    return root / name / "inputs"


def bundle(inputs):
    return read_pickle(inputs / "inputs_bundle.pkl")


def differing(a: dict, b: dict) -> list[str]:
    return [k for k in SOLVER_INPUTS if a["exact"][k] != b["exact"][k]]


@pytest.fixture(scope="module")
def city(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2city")
    data = sv.base(root)
    inputs = build(data, root, "census")
    buildings = gpd.read_file(inputs / "buildings_alkis_allocated.gpkg")
    graph = read_pickle(inputs / "walk_graph_base.pkl")
    return {
        "root": root,
        "data": data,
        "inputs": inputs,
        "bundle": bundle(inputs),
        "buildings": buildings,
        "graph": graph,
        "districts": bundle(inputs)["districts"],
        "fingerprint": fingerprint_inputs(inputs),
    }


def direct_spec(path, geometry="points", **extra) -> dict:
    return {
        "geometry": geometry,
        "file": str(path),
        "id_column": "id",
        "children_0_5": "children",
        "count_type": "observed",
        "reference_year": 2024,
        "source": "test",
        "district_column": "",
        **extra,
    }


def save(frame: gpd.GeoDataFrame, path):
    frame.to_file(path, driver="GPKG")
    return path


def with_direct(data: dict, spec: dict) -> dict:
    data = copy.deepcopy(data)
    data["demand"] = {"mode": "direct", "direct": spec}
    return data


# ------------------------------------------------------------------------------------------------ census_hybrid
def test_census_hybrid_v2_equals_the_version_1_build_except_the_label(city):
    assert not differing(city["fingerprint"], GOLDEN) and all(
        city["fingerprint"]["summary"][k] == v for k, v in GOLDEN["summary"].items()
    )
    assert city["fingerprint"]["exact"]["labels"] != GOLDEN["exact"]["labels"]  # label only
    assert set(city["bundle"]["demand"]["demand_definition"]) == {
        "projected 2030 children aged 0-5, census-grid allocation (census 2022 cell shares, footprint within cells)"
    }
    report = json.loads((city["inputs"] / "validation_report.json").read_text(encoding="utf-8"))
    assert report["methods"]["demand"] == {
        "mode": "census_hybrid",
        "count_type": "projected",
        "reference_year": 2030,
        "source": "synthetic forecast (make_city.py)",
        "census_year": 2022,
    }
    origins = pd.read_csv(city["inputs"] / "demand_origins.csv")
    assert set(origins["origin_type"]) == {"building", "census_cell_centre"}


# ------------------------------------------------------------------------------------------------ building_area
def test_building_area_equals_supplied_district_polygons_allocated_by_footprint(city):
    root, data = city["root"], copy.deepcopy(city["data"])
    data["demand"] = {"mode": "building_area"}
    footprint = build(data, root, "footprint")
    districts = city["districts"].to_crs(25832)
    polygons = districts[["district_id", "geometry"]].rename(columns={"district_id": "id"})
    totals = city["bundle"]["children"].set_index("district_id")
    polygons["children"] = polygons["id"].map(totals["children_0_5"]).astype(float)
    polygons["u3"], polygons["k35"] = (
        polygons["id"].map(totals["age_0_2"]).astype(float),
        polygons["id"].map(totals["age_3_5"]).astype(float),
    )
    spec = direct_spec(
        save(polygons, root / "districts_as_polygons.gpkg"),
        "polygons",
        polygon_allocation="building_area",
        boundary_rule="reject",
        age_0_2="u3",
        age_3_5="k35",
    )
    supplied = build(with_direct(city["data"], spec), root, "district_polygons")
    a, b = (
        bundle(footprint)["demand"].set_index("demand_id"),
        bundle(supplied)["demand"].set_index("demand_id"),
    )
    assert a[["nearest_node", "district_id"]].equals(b[["nearest_node", "district_id"]])
    for column in ["estimated_demand", "age_0_2", "age_3_5"]:
        assert np.allclose(a[column], b[column], rtol=0, atol=1e-12)
    assert np.isclose(a["estimated_demand"].sum(), sum(sum(v) for v in CHILDREN.values()))
    assert set(bundle(footprint)["demand"]["demand_definition"]) == {
        "projected 2030 children aged 0-5, footprint allocation (footprint shares within districts)"
    }
    assert set(pd.read_csv(supplied / "demand_origins.csv")["origin_type"]) == {"polygon_building"}


def test_wide_observed_totals_without_an_age_split(city):
    root, data = city["root"], copy.deepcopy(city["data"])
    table = pd.DataFrame({"district": [str(k) for k in CHILDREN], "kids": [float(sum(v)) for v in CHILDREN.values()]})
    table.to_csv(root / "children_wide.csv", index=False)
    data["demand"] = {"mode": "building_area"}
    data["population"] = {
        "file": str(root / "children_wide.csv"),
        "layout": "wide",
        "district_id_column": "district",
        "count_type": "observed",
        "reference_year": 2024,
        "source": "test register",
        "children_0_5": "kids",
    }
    inputs = build(data, root, "wide")
    b = bundle(inputs)
    assert b["demand"][["age_0_2", "age_3_5"]].isna().all().all()  # no split invented
    assert "age_0_2" not in b["children"] and np.isclose(b["demand"]["estimated_demand"].sum(), 75.0)
    assert set(b["demand"]["demand_definition"]) == {
        "observed 2024 children aged 0-5, footprint allocation (footprint shares within districts)"
    }


# ------------------------------------------------------------------------------------------------ direct
def test_direct_points_from_the_exported_origins_reproduce_the_census_hybrid_inputs(city):
    root = city["root"]
    origins = pd.read_csv(city["inputs"] / "demand_origins.csv", float_precision="round_trip")  # exact floats
    points = gpd.GeoDataFrame(
        {
            "id": origins["origin_id"].astype(str),
            "children": origins["children_0_5"],
            "u3": origins["age_0_2"],
            "k35": origins["age_3_5"],
        },
        geometry=gpd.points_from_xy(origins["x"], origins["y"]),
        crs=25832,
    )
    spec = direct_spec(save(points, root / "points.gpkg"), age_0_2="u3", age_3_5="k35")
    inputs = build(with_direct(city["data"], spec), root, "points")
    fresh = fingerprint_inputs(inputs)
    # exact for every model input; the district table now holds the sums of the supplied points (floats, not the
    # forecast's integers) and is compared numerically
    assert differing(fresh, city["fingerprint"]) == ["children"]
    a, b = (
        bundle(inputs)["children"].set_index("district_id"),
        city["bundle"]["children"].set_index("district_id"),
    )
    assert np.allclose(
        a[["children_0_5", "age_0_2", "age_3_5"]],
        b.loc[a.index, ["children_0_5", "age_0_2", "age_3_5"]],
        rtol=0,
        atol=1e-9,
    )
    assert set(bundle(inputs)["demand"]["demand_definition"]) == {"observed 2024 children aged 0-5, supplied points"}
    assert set(pd.read_csv(inputs / "demand_origins.csv")["origin_type"]) == {"supplied_point"}


def test_direct_polygons_by_network_nodes_conserve_totals_and_are_deterministic(city):
    root = city["root"]
    polygons = gpd.GeoDataFrame(
        {"id": ["a", "b"], "children": [10.0, 6.0]},
        geometry=[
            box(X0 + 50, Y0 + 50, X0 + 250, Y0 + 250),
            box(X0 + 350, Y0 + 350, X0 + 550, Y0 + 550),
        ],
        crs=25832,
    )
    spec = direct_spec(
        save(polygons, root / "areas.gpkg"),
        "polygons",
        polygon_allocation="network_nodes",
        boundary_rule="reject",
    )
    first, second = (
        build(with_direct(city["data"], spec), root, "nodes1"),
        build(with_direct(city["data"], spec), root, "nodes2"),
    )
    assert not differing(fingerprint_inputs(first), fingerprint_inputs(second))
    origins = pd.read_csv(first / "demand_origins.csv")
    assert origins.groupby("source_id")["children_0_5"].sum().to_dict() == pytest.approx(
        {"a": 10.0, "b": 6.0}, abs=1e-12
    )
    assert set(origins["origin_type"]) == {"polygon_node"} and (origins.groupby("source_id").size() == 4).all()
    assert np.isclose(bundle(first)["demand"]["estimated_demand"].sum(), 16.0)


# ------------------------------------------------------------------------------------------------ rejected inputs
def run_direct(city, frame, geometry="points", **extra):
    path = city["root"] / f"case_{abs(hash(str(frame.to_json())))}.gpkg"
    spec = direct_spec(save(frame, path) if frame.crs else path, geometry, **extra)
    if frame.crs is None:
        frame.to_file(path, driver="GPKG")
    return dm.direct(spec, city["districts"], city["graph"], city["buildings"])


def points(rows, crs=25832, **columns):
    return gpd.GeoDataFrame({"id": [str(i) for i in range(len(rows))], **columns}, geometry=rows, crs=crs)


def test_points_outside_districts_disagreeing_districts_and_bad_counts_are_rejected(city):
    inside = Point(X0 + 120, Y0 + 120)
    with pytest.raises(dm.DemandError, match="outside every district"):
        run_direct(city, points([inside, Point(X0 - 50, Y0 - 50)], children=[1.0, 2.0]))
    with pytest.raises(dm.DemandError, match="missing, non-numeric or negative"):
        run_direct(city, points([inside], children=[-1.0]))
    with pytest.raises(dm.DemandError, match="missing, non-numeric or negative"):
        run_direct(city, points([inside], children=[None]))
    with pytest.raises(dm.DemandError, match="age_0_2 \\+ age_3_5 differs"):
        run_direct(city, points([inside], children=[5.0], u3=[1.0], k35=[1.0]), age_0_2="u3", age_3_5="k35")
    with pytest.raises(dm.DemandError, match="disagrees with the location"):
        run_direct(city, points([inside], children=[1.0], district=["2"]), district_column="district")


def test_ambiguous_or_unsupported_geometries_are_rejected_not_converted(city):
    with pytest.raises(dm.DemandError, match="allows \\['Point'\\], found \\['MultiPoint'\\]"):
        run_direct(city, points([MultiPoint([(X0 + 120, Y0 + 120), (X0 + 130, Y0 + 130)])], children=[1.0]))
    with pytest.raises(dm.DemandError, match="found \\['Point', 'Polygon'\\]"):
        run_direct(
            city,
            points(
                [box(X0 + 50, Y0 + 50, X0 + 150, Y0 + 150), Point(X0 + 120, Y0 + 120)],
                children=[1.0, 1.0],
            ),
            "polygons",
            polygon_allocation="building_area",
            boundary_rule="reject",
        )
    with pytest.raises(dm.DemandError, match="overlapping polygons"):
        run_direct(
            city,
            points(
                [
                    box(X0 + 50, Y0 + 50, X0 + 250, Y0 + 250),
                    box(X0 + 150, Y0 + 150, X0 + 280, Y0 + 280),
                ],
                children=[1.0, 1.0],
            ),
            "polygons",
            polygon_allocation="building_area",
            boundary_rule="reject",
        )
    no_crs = points([Point(X0 + 120, Y0 + 120)], crs=None, children=[1.0])
    with pytest.raises(dm.DemandError, match="no CRS"):
        run_direct(city, no_crs)


def test_polygons_crossing_districts_are_rejected_or_split_explicitly(city):
    crossing = points([box(X0 + 150, Y0 + 50, X0 + 450, Y0 + 250)], children=[12.0])
    with pytest.raises(dm.DemandError, match="cross district boundaries \\(boundary_rule = 'reject'\\)"):
        run_direct(city, crossing, "polygons", polygon_allocation="building_area", boundary_rule="reject")
    origins, audit = run_direct(city, crossing, "polygons", polygon_allocation="building_area", boundary_rule="split")
    assert audit["crossing_district_boundaries"] == 1 and origins["district_id"].nunique() == 2
    assert np.isclose(origins["children_0_5"].sum(), 12.0, rtol=0, atol=1e-12)


def test_polygons_without_allocation_locations_are_rejected(city):
    empty = points([box(X0 + 10, Y0 + 10, X0 + 60, Y0 + 60)], children=[3.0])
    with pytest.raises(dm.DemandError, match="contain no eligible residential buildings"):
        run_direct(city, empty, "polygons", polygon_allocation="building_area", boundary_rule="reject")
    with pytest.raises(dm.DemandError, match="contain no walk-network nodes"):
        run_direct(city, empty, "polygons", polygon_allocation="network_nodes", boundary_rule="reject")


def test_input_crs_is_converted_not_assumed(city):
    projected = points([Point(X0 + 120, Y0 + 120), Point(X0 + 420, Y0 + 480)], children=[1.0, 2.0])
    a, _ = run_direct(city, projected)
    b, _ = run_direct(city, projected.to_crs(4326))
    assert a["district_id"].tolist() == b["district_id"].tolist()
    assert np.allclose(a[["x", "y"]], b[["x", "y"]], atol=1e-6)


def test_wide_totals_reject_missing_negative_inconsistent_and_unknown_districts(city):
    root, districts = city["root"], city["districts"]
    cases = {
        "missing": (
            {"d": ["1", "2", "3", "4"], "kids": [1, None, 3, 4]},
            "missing, non-numeric or negative",
        ),
        "negative": (
            {"d": ["1", "2", "3", "4"], "kids": [1, -2, 3, 4]},
            "missing, non-numeric or negative",
        ),
        "ages": (
            {
                "d": ["1", "2", "3", "4"],
                "kids": [3, 3, 3, 3],
                "u3": [1, 1, 1, 1],
                "k35": [1, 1, 1, 1],
            },
            "differs from children_0_5",
        ),
        "unknown": (
            {"d": ["1", "2", "3", "9"], "kids": [1, 2, 3, 4]},
            "district_id not in the districts",
        ),
    }
    for name, (table, message) in cases.items():
        pd.DataFrame(table).to_csv(root / f"{name}.csv", index=False)
        spec = {
            "file": str(root / f"{name}.csv"),
            "district_id_column": "d",
            "children_0_5": "kids",
            **({"age_0_2": "u3", "age_3_5": "k35"} if name == "ages" else {}),
        }
        with pytest.raises(ValueError, match=message):
            dm.child_totals(spec, districts, 0)


def test_consistency_statuses_separate_missing_coverage_from_differences():
    population = pd.DataFrame({"district_id": ["1", "2", "3", "4"], "children_0_5": [10.0, 20.0, 5.0, 0.0]})
    direct = pd.Series({"1": 10.0, "2": 18.0, "4": 0.0, "9": 3.0})
    table = dm.consistency(direct, population, tolerance=0.5).set_index("district_id")
    assert table["status"].to_dict() == {
        "1": "match",
        "2": "differs",
        "3": "no_direct_demand",
        "4": "match",
        "9": "no_district_total",
    }
    assert table.loc["2", "abs_difference"] == 2.0 and table.loc["2", "pct_difference"] == -10.0
    assert np.isnan(table.loc["4", "pct_difference"])  # no percentage of a zero total


def test_direct_demand_against_district_totals_report_or_error(city):
    root = city["root"]
    # Suedwest (3) matches its total 12, Nordwest (1) has 15, Nordost (2) 25 instead of 27, Suedost (4) has no point
    nordost = [Point(X0 + x, Y0 + y) for x, y in [(410, 410), (510, 410), (410, 510), (510, 510)]]
    sites = points(
        [Point(X0 + 120, Y0 + 120), Point(X0 + 120, Y0 + 480), *nordost],
        children=[12.0, 15.0, 7.0, 6.0, 6.0, 6.0],
    )
    spec = direct_spec(save(sites, root / "partial.gpkg"))
    inputs = build(with_direct(city["data"], spec), root, "consistency_report")  # default policy: report
    table = pd.read_csv(inputs / "direct_vs_district_totals.csv", dtype={"district_id": str}).set_index("district_id")
    assert table["status"].to_dict() == {
        "1": "match",
        "2": "differs",
        "3": "match",
        "4": "no_direct_demand",
    }
    assert table.loc["2", "abs_difference"] == 2.0 and np.isclose(table.loc["2", "pct_difference"], -200 / 27)
    assert np.isclose(bundle(inputs)["demand"]["estimated_demand"].sum(), 52.0)  # kept as supplied, not rescaled
    config = sv.write(
        with_direct(city["data"], {**spec, "consistency": "error", "consistency_tolerance": 0.5}),
        root / "strict.toml",
    )
    run = cli(config, "build", "--workspace", str(root / "consistency_error"))
    assert run.returncode != 0 and "policy 'error'" in run.stderr and "no_direct_demand" in run.stderr
    bad = sv.write(with_direct(city["data"], {**spec, "consistency": "rescale"}), root / "bad_policy.toml")
    assert "consistency] must be one of ['report', 'error']" in cli(bad, "check").stdout


def test_check_reports_missing_configured_columns_before_building(city):
    root, data = city["root"], copy.deepcopy(city["data"])
    pd.DataFrame({"district": ["1", "2", "3", "4"], "kids": [1, 2, 3, 4]}).to_csv(root / "cols.csv", index=False)
    data["demand"] = {"mode": "building_area"}
    data["population"] = {
        "file": str(root / "cols.csv"),
        "layout": "wide",
        "district_id_column": "district_code",
        "count_type": "observed",
        "reference_year": 2024,
        "source": "test",
        "children_0_5": "children",
    }
    run = cli(sv.write(data, root / "cols.toml"), "check")
    assert run.returncode != 0 and "lacks configured column(s) ['district_code', 'children']" in run.stdout
    spec = direct_spec(save(points([Point(X0 + 120, Y0 + 120)], kids=[1.0]), root / "cols.gpkg"))
    run = cli(sv.write(with_direct(city["data"], spec), root / "cols_direct.toml"), "check")
    assert run.returncode != 0 and "[demand.direct]" in run.stdout and "['children']" in run.stdout
