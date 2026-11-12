"""The same standardised inputs give the same optimisation results whatever the adapter (synthetic city).

Four routes to the inputs of the version-2 census-grid build are solved for P = 1, 2, 3 and compared with it, model by
model: formulation size (rows, columns, nonzeros), solver status, certification, objective, best bound and gap. The
sites must be the same network nodes; different sites are accepted only as an equivalent certified optimum (same
objective, both proven optimal), since optima need not be unique.
"""

from __future__ import annotations

import copy

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import synthetic_schema2 as sv
from test_workflow_synthetic import _NoSites, cli

from childcare_access import solutions as sol
from childcare_access.script_support import read_pickle

pytestmark = pytest.mark.gurobi
FORMULATION = ["rows", "columns", "nonzeros"]


def build(config, workspace):
    run = cli(config, "build", "--workspace", str(workspace))
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    return workspace / "inputs"


def solve_all(config, workspace) -> dict:
    out = {}
    for P in (1, 2, 3):
        run = cli(config, "solve", "--workspace", str(workspace), "--p", str(P), "--run-name", f"P{P}")
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        folder = workspace / "runs" / f"P{P}"
        records = pd.concat([pd.read_csv(f) for f in sorted(folder.glob("*/*_solver_records.csv"))]).set_index("role")
        nodes = read_pickle(workspace / "inputs" / "inputs_bundle.pkl")["candidates"].set_index("facility_id")[
            "nearest_node"
        ]
        out[P] = {
            "records": records,
            "solutions": sol.load_reported_solutions(run_dir=folder, config=_NoSites(), require_all=False),
            "nodes": nodes,
        }
    return out


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    root = tmp_path_factory.mktemp("equivalence")
    base = sv.base(root)
    configs = {
        "census": sv.write(base, root / "census.toml"),
        "legacy": root / "synthetic_city.toml",
    }
    inputs = build(configs["census"], root / "census")
    # supplied points: the exported census-grid allocation origins
    origins = pd.read_csv(inputs / "demand_origins.csv", float_precision="round_trip")
    gpd.GeoDataFrame(
        {
            "id": origins["origin_id"].astype(str),
            "children": origins["children_0_5"],
            "u3": origins["age_0_2"],
            "k35": origins["age_3_5"],
        },
        geometry=gpd.points_from_xy(origins["x"], origins["y"]),
        crs=25832,
    ).to_file(root / "points.gpkg", driver="GPKG")
    points = copy.deepcopy(base)
    points["demand"] = {
        "mode": "direct",
        "direct": {
            "geometry": "points",
            "file": str(root / "points.gpkg"),
            "id_column": "id",
            "children_0_5": "children",
            "age_0_2": "u3",
            "age_3_5": "k35",
            "count_type": "projected",
            "reference_year": 2030,
            "source": "exported origins",
            "district_column": "",
        },
    }
    configs["points"] = sv.write(points, root / "points.toml")
    # external index: the four-indicator values themselves
    index = read_pickle(inputs / "inputs_bundle.pkl")["districts"].set_index("district_id")[
        "childcare_vulnerability_index"
    ]
    pd.DataFrame({"d": index.index, "v": index.values}).to_csv(root / "index.csv", index=False)
    external = copy.deepcopy(base)
    external["vulnerability"] = {
        "mode": "external_index",
        "external_index": {
            "file": str(root / "index.csv"),
            "district_id_column": "d",
            "value_column": "v",
            "name": "four-indicator values",
            "definition": "test",
            "higher_is_more_vulnerable": True,
            "quantile": base["vulnerability"]["four_indicator"]["quantile"],
            "ties": "include",
            "missing": "error",
        },
    }
    configs["external"] = sv.write(external, root / "external.toml")
    # supplied candidate sites at the residential building locations
    buildings = gpd.read_file(inputs / "buildings_alkis_allocated.gpkg")
    gpd.GeoDataFrame(
        {"site": buildings["building_id"].astype(str)},
        geometry=gpd.points_from_xy(buildings["rep_x"], buildings["rep_y"]),
        crs=25832,
    ).to_file(root / "sites.gpkg", driver="GPKG")
    sites = copy.deepcopy(base)
    sites["candidates"] = {"source": "file", "file": str(root / "sites.gpkg"), "id_column": "site"}
    configs["sites"] = sv.write(sites, root / "sites.toml")

    results = {}
    for name, config in configs.items():
        if name != "census":
            build(config, root / name)
        results[name] = solve_all(config, root / name)
    return results


@pytest.mark.parametrize("route", ["legacy", "points", "external", "sites"])
def test_every_adapter_gives_the_same_certified_results(routes, route):
    for P in (1, 2, 3):
        a, b = routes["census"][P], routes[route][P]
        assert list(a["records"].index) == list(b["records"].index), (route, P)
        for role in a["records"].index:  # same formulation, status, objective, bound and gap
            ra, rb = a["records"].loc[role], b["records"].loc[role]
            assert [ra[c] for c in FORMULATION] == [rb[c] for c in FORMULATION], (route, P, role)
            assert ra["status"] == rb["status"], (route, P, role)
            for column in ("incumbent_objective", "best_bound", "mip_gap"):
                assert np.isclose(ra[column], rb[column], rtol=1e-9, atol=1e-9, equal_nan=True), (
                    route,
                    P,
                    role,
                    column,
                )
        for key, sa in a["solutions"].items():
            sb = b["solutions"][key]
            assert sa.certification == sb.certification and np.isclose(
                sa.objective_value, sb.objective_value, rtol=1e-12, atol=1e-12
            )
            same = sorted(a["nodes"][list(sa.sites)]) == sorted(b["nodes"][list(sb.sites)])
            assert same or (sa.certification == sol.PROVEN == sb.certification), (
                route,
                P,
                key,
                "different sites, not both proven",
            )
            if not same:
                print(f"{route} P={P} {key}: other sites, same proven optimum (alternative optimum)")
