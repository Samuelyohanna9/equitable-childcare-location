"""Provenance and classified comparisons on the synthetic city."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest
import synthetic_schema2 as sv
from test_workflow_synthetic import cli

from childcare_access import comparison as cmp
from childcare_access import provenance as prov


def build(data: dict, root, name: str):
    run = cli(sv.write(data, root / f"{name}.toml"), "build", "--workspace", str(root / name))
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    return root / name / "inputs"


def solve(config, workspace, P: int):
    run = cli(config, "solve", "--workspace", str(workspace), "--p", str(P), "--run-name", f"P{P}")
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    return workspace / "runs" / f"P{P}"


@pytest.fixture(scope="module")
def city(tmp_path_factory):
    root = tmp_path_factory.mktemp("provenance")
    data = sv.base(root)  # also writes the version-1 file root/synthetic_city.toml
    legacy = cli(root / "synthetic_city.toml", "build", "--workspace", str(root / "legacy"))
    assert legacy.returncode == 0, legacy.stderr[-3000:]
    return {
        "root": root,
        "data": data,
        "census": build(data, root, "census"),
        "again": build(data, root, "again"),
        "legacy": root / "legacy" / "inputs",
    }


def variant(city, name, **tables):
    data = copy.deepcopy(city["data"])
    data.update(tables)
    return build(data, city["root"], name)


# ------------------------------------------------------------------------------------------------ provenance
def test_provenance_is_complete_and_deterministic(city):
    a, b = prov.load(city["census"]), prov.load(city["again"])
    assert {
        "schema",
        "city",
        "configuration",
        "inputs",
        "demand",
        "vulnerability",
        "candidates",
        "facilities",
        "network",
        "crs",
        "parameters",
        "software",
        "identity",
        "redacted",
    } <= set(a) and "incomplete" not in a
    assert a["configuration"]["schema_version"] == 2 and a["city"]["scenario"] == "synthetic_city"
    assert (
        a["demand"]["mode"] == "census_hybrid"
        and a["demand"]["census_year"] == 2022
        and a["demand"]["count_type"] == "projected"
    )
    assert (
        a["vulnerability"]["selected"] == ["Nordost", "Suedost"]
        and a["candidates"]["source"] == "residential_buildings"
    )
    assert a["network"]["rule_set"] == "pedestrian_access_v3" and a["crs"] == {
        "projected": 25832,
        "census": 3035,
    }
    assert a["redacted"] == [] and "python" in a["software"]

    # same configuration and data: the same record, except the file bytes (pickles are not byte-reproducible)
    def strip(record: dict) -> dict:  # the two test configurations differ in name only
        return {
            **record,
            "identity": {k: v for k, v in record["identity"].items() if k != "file_bytes"},
            "configuration": {**record["configuration"], "file": None},
        }

    assert strip(a) == strip(b)
    assert prov.load(city["legacy"]) is None  # version-1 builds have none


def test_secrets_never_reach_provenance_or_reports(city):
    data = copy.deepcopy(city["data"])
    data["osm"]["api_key"] = "s3cr3t-api-value"
    data["population"]["source"] = "forecast; Set parameter LicenseID to value 8675309-fake"
    inputs = build(data, city["root"], "secrets")
    for name in ("provenance.json", "validation_report.json"):
        text = (inputs / name).read_text(encoding="utf-8")
        assert "s3cr3t-api-value" not in text and "8675309-fake" not in text, name
    record = prov.load(inputs)
    assert set(record["redacted"]) >= {
        "configuration.settings.osm.api_key",
        "configuration.settings.population.source",
    }
    assert prov.redact({"password": "x", "nested": {"token": "y"}, "ok": "plain"}) == {
        "password": "<redacted>",
        "nested": {"token": "<redacted>"},
        "ok": "plain",
    }


def test_incomplete_provenance_is_reported(city, tmp_path):
    damaged = tmp_path / "damaged"
    damaged.mkdir()
    (damaged / "provenance.json").write_text(json.dumps({"schema": prov.SCHEMA, "city": {}}), encoding="utf-8")
    assert set(prov.load(damaged)["incomplete"]) >= {"demand", "vulnerability", "identity"}
    assert cmp.assumptions(city["legacy"])["provenance"].startswith("INCOMPLETE: no provenance record")


# ------------------------------------------------------------------------------------------------ classification
def test_identical_inputs_through_different_configurations_are_class_a(city):
    report = cmp.compare_runs({"inputs": city["legacy"]}, {"inputs": city["census"]})
    c = report["classification"]
    assert (
        c["class"] == "A" and c["differing_components"] == [] and not c["labels_identical"]
    )  # label only ("census 2022")
    assert [d["aspect"] for d in report["assumption_differences"]] == ["demand definition"]


def test_same_city_with_other_assumptions_is_a_sensitivity_analysis(city):
    root = city["root"]
    pd.DataFrame({"d": ["1", "4"]}).to_csv(root / "listed.csv", index=False)
    cases = {
        "footprint": ({"demand": {"mode": "building_area"}}, {"demand definition"}),
        "listed": (
            {
                "vulnerability": {
                    "mode": "priority_districts",
                    "priority_districts": {
                        "file": str(root / "listed.csv"),
                        "district_id_column": "d",
                        "label": "test list",
                        "definition": "test",
                    },
                }
            },
            {"priority group", "priority districts"},
        ),
        "nodes": ({"candidates": {"source": "demand_nodes"}}, {"candidates"}),
    }
    for name, (tables, expected) in cases.items():
        report = cmp.compare_runs({"inputs": city["census"]}, {"inputs": variant(city, name, **tables)})
        assert report["classification"]["class"] == "B" and report["classification"]["same_district_geometry"], name
        assert expected <= {d["aspect"] for d in report["assumption_differences"]}, name
        assert "sensitivity analysis" in cmp.to_markdown(report)


def test_different_cities_are_described_not_compared():
    a = {
        "exact": {"district_geometry": "x", "demand": "1", "labels": "l"},
        "file_sha256": {"f": "1"},
    }
    b = {
        "exact": {"district_geometry": "y", "demand": "1", "labels": "l"},
        "file_sha256": {"f": "1"},
    }
    assert cmp.classify(a, b)["class"] == "C"  # equal content elsewhere does not matter


def test_the_cli_writes_a_classified_report(city):
    run = cli(
        city["root"] / "census.toml",
        "compare",
        "--workspace",
        str(city["root"] / "census"),
        "--against-inputs",
        str(city["legacy"]),
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert "class A" in (city["root"] / "census" / "compare" / "classified_1" / "comparison.md").read_text(
        encoding="utf-8"
    )


# ------------------------------------------------------------------------------------------------ solver status
def test_certification_categories_are_taken_from_the_records(tmp_path):
    folder = tmp_path / "run" / "utilitarian"
    folder.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "role": "proven",
                "status": "Optimal",
                "incumbent_objective": 5.0,
                "best_bound": 5.0,
                "mip_gap": 0.0,
                "runtime_seconds": 1,
                "time_limit_seconds": 60,
            },
            {
                "role": "within gap",
                "status": "Optimal",
                "incumbent_objective": 5.0,
                "best_bound": 4.98,
                "mip_gap": 0.004,
                "runtime_seconds": 1,
                "time_limit_seconds": 60,
            },
            {
                "role": "time limit",
                "status": "TimeLimit",
                "incumbent_objective": 6.0,
                "best_bound": 2.0,
                "mip_gap": 0.67,
                "runtime_seconds": 60,
                "time_limit_seconds": 60,
            },
            {
                "role": "infeasible",
                "status": "Infeasible",
                "incumbent_objective": None,
                "best_bound": None,
                "mip_gap": None,
                "runtime_seconds": 1,
                "time_limit_seconds": 60,
            },
        ]
    ).to_csv(folder / "utilitarian_solver_records.csv", index=False)
    table = cmp.status_table(tmp_path / "run").set_index("role")
    assert table.loc["proven", "certification"] == "proven optimal"
    assert table.loc["within gap", "certification"].startswith(
        "feasible incumbent within the requested MIP gap, NOT proven"
    )
    assert table.loc["time limit", "certification"] == "time-limited: feasible incumbent, NOT proven optimal"
    assert table.loc["infeasible", "certification"].startswith("no solution (Infeasible")
    assert pd.isna(table.loc["infeasible", "objective"]) and pd.isna(
        table.loc["infeasible", "mip_gap"]
    )  # nothing invented


@pytest.mark.gurobi
def test_runs_on_identical_inputs_are_compared_directly_and_a_different_p_is_flagged(city):
    root = city["root"]
    legacy = solve(root / "synthetic_city.toml", root / "legacy", 2)
    census = solve(root / "census.toml", root / "census", 2)
    report = cmp.compare_runs({"inputs": city["legacy"], "run": legacy}, {"inputs": city["census"], "run": census})
    assert report["classification"]["class"] == "A"
    for key, d in report["differences"].items():
        assert abs(d["objective_difference"]) <= 1e-9 and d["certification"][0] == d["certification"][1], key
    assert {"Q4 mean", "Q4 demand-weighted mean"} <= set(
        report["runs"]["a"]["results"]["utilitarian"]["indicators"]
    ) or {"Priority group mean", "Priority group demand-weighted mean"} <= set(
        report["runs"]["a"]["results"]["utilitarian"]["indicators"]
    )
    three = solve(root / "census.toml", root / "census", 3)
    flagged = cmp.compare_runs({"inputs": city["census"], "run": census}, {"inputs": city["census"], "run": three})
    assert flagged["classification"]["class"] == "B" and "different P" in flagged["classification"]["meaning"]
    assert {"aspect": "P", "a": 2, "b": 3} in flagged["assumption_differences"]
