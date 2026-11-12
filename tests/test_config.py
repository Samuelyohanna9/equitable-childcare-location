"""City configuration.

Invalid configurations fail with clear messages, the
reusable modules hold no Münster-specific literals, and a synthetic second city runs through the same loaders by
configuration alone (separate process, CHILDCARE_ACCESS_CONFIG).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from childcare_access import config as cfg

MUENSTER = cfg.PROJECT_ROOT / "config" / "muenster.toml"
REUSABLE = [
    "config",
    "settings",
    "demand",
    "vulnerability",
    "network",
    "optimisation",
    "validation",
    "evaluation",
    "solutions",
    "workflow",
    "adapters",
    "pedestrian_network",
    "all_demand_candidates",
    "district_demand_maximin",
    "saved_inputs",
]


def write_config(tmp_path: Path, data: dict) -> Path:
    """Write a TOML file (minimal writer for the value types used in city configurations)."""

    def value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return repr(v)

    lines = []

    def table(prefix: str, node: dict) -> None:
        scalars = {k: v for k, v in node.items() if not isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        lines.extend(f"{json.dumps(k)} = {value(v)}" for k, v in scalars.items())
        for k, v in node.items():
            if isinstance(v, dict):
                table(f"{prefix}.{k}" if prefix else k, v)

    table("", data)
    path = tmp_path / "city.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def muenster_data() -> dict:
    return tomllib.loads(MUENSTER.read_text(encoding="utf-8"))


def expect_error(tmp_path: Path, data: dict, fragment: str) -> None:
    with pytest.raises(cfg.ConfigError) as error:
        cfg.load_city_config(write_config(tmp_path, data))
    assert fragment in str(error.value), str(error.value)


# ----------------------------------------------------------------------------------------- validation messages
def test_missing_configuration_file(tmp_path):
    with pytest.raises(cfg.ConfigError, match="city configuration not found"):
        cfg.load_city_config(tmp_path / "nowhere.toml")


def test_missing_key_is_named(tmp_path):
    data = muenster_data()
    del data["sources"]["forecast"]["value_column"]
    expect_error(tmp_path, data, "missing key [sources.forecast] value_column")


@pytest.mark.parametrize(
    "code, fragment",
    [
        (999999, "is not a valid CRS"),
        (4326, "is not a projected CRS"),
        (2263, "does not use metres"),
    ],
)
def test_invalid_crs(tmp_path, code, fragment):
    data = muenster_data()
    data["crs"]["projected"] = code
    expect_error(tmp_path, data, fragment)


def test_incompatible_parameters(tmp_path):
    data = muenster_data()
    data["sources"]["dormitories"]["report_min_share"] = 0.95
    data["sources"]["forecast"]["district_code_pattern"] = r"^\d{2}\s"
    data["model_parameters"]["least_advantaged_quantile"] = 1.5
    with pytest.raises(cfg.ConfigError) as error:
        cfg.load_city_config(write_config(tmp_path, data))
    message = str(error.value)
    assert (
        "3 problem(s)" in message
        and "report_min_share" in message
        and "capture group" in message
        and "strictly between" in message
    )


def test_the_muenster_parameters_equal_the_defaults():
    config = cfg.load_city_config(MUENSTER)
    assert config.parameters == cfg.PARAMETER_DEFAULTS


def test_the_parameters_may_be_changed(tmp_path):
    data = muenster_data()
    data["model_parameters"].update(facility_budget=5, walking_speed_m_per_min=75.0, least_advantaged_quantile=0.8)
    data["solver_resources"]["threads"] = 8
    config = cfg.load_city_config(write_config(tmp_path, data))
    assert config.parameters["facility_budget"] == 5 and config.parameters["walking_speed_m_per_min"] == 75.0
    assert config.parameters["least_advantaged_quantile"] == 0.8 and config.parameters["time_limit_seconds"] == 900


def test_the_parameters_are_checked(tmp_path):
    data = muenster_data()
    data["model_parameters"].update(facility_budget=0, walk_speed=5)
    with pytest.raises(cfg.ConfigError) as error:
        cfg.load_city_config(write_config(tmp_path, data))
    assert "unknown parameter(s) ['walk_speed']" in str(
        error.value
    ) and "facility_budget must be an integer >= 1" in str(error.value)


def test_missing_dataset_and_missing_columns_are_reported(tmp_path):
    data = muenster_data()
    forecast = tmp_path / "forecast.csv"
    forecast.write_text("RAUM;ZEIT;WERT\n01 A;2026;1\n", encoding="utf-8")
    data["sources"]["forecast"]["file"] = str(forecast)
    data["sources"]["social"]["file"] = str(tmp_path / "absent.csv")
    with pytest.raises(cfg.ConfigError) as error:
        cfg.check_data(cfg.load_city_config(write_config(tmp_path, data)), checksums=False)
    message = str(error.value)
    assert "missing dataset [sources.social.file]" in message
    assert "lacks configured column(s) ['GESCHLECHT', 'ALTERSGRUPPE']" in message


# ------------------------------------------------------------------------- reusable code holds no Münster values
MUENSTER_Q4 = [  # the 12 Q4 districts of the case study
    "Angelmodde",
    "Bahnhof",
    "Berg Fidel",
    "Coerde",
    "Geist",
    "Gievenbeck",
    "Gremmendorf-West",
    "Hiltrup-Mitte",
    "Kinderhaus-Ost",
    "Kinderhaus-West",
    "Neutor",
    "Schützenhof",
]


def test_reusable_modules_hold_no_muenster_literals():
    """No Münster names, district names, totals, district count or CRS codes in code (docstrings excluded)."""
    names = [*MUENSTER_Q4, "Uppenberg"]
    words = re.compile(r"m(ü|ue|u)nster(?!\.toml)|05515000|\b(" + "|".join(map(re.escape, names)) + r")\b", re.I)
    numbers = {14960, 14965, 7322, 7638, 45, 25832, 3035}
    hits = []
    for module in REUSABLE:
        path = cfg.PROJECT_ROOT / "src" / "childcare_access" / f"{module}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and id(node) not in docstrings:
                if isinstance(node.value, str) and words.search(node.value):
                    hits.append(f"{module}.py:{node.lineno} {node.value[:60]!r}")
                elif type(node.value) is int and node.value in numbers:
                    hits.append(f"{module}.py:{node.lineno} {node.value}")
    assert not hits, "Münster-specific literals in reusable code (move them to config/<city>.toml):\n" + "\n".join(hits)


# ----------------------------------------------------------------- a second city through configuration alone
def _synthetic_city(tmp_path: Path) -> Path:
    """Three districts in EPSG:25833 with other column names, 3-digit IDs, comma CSVs and other years."""
    squares = [box(390000 + 1000 * k, 5810000, 391000 + 1000 * k, 5811000) for k in range(3)]
    gpd.GeoDataFrame(
        {"bez_nr": [1, 2, 3], "bez_name": ["Nord", "Mitte-Ost", "Sued"]},
        geometry=squares,
        crs=25833,
    ).to_file(tmp_path / "districts.gpkg", driver="GPKG")
    rows = []
    for nr, name, young, older in [
        (1, "Nord", 30, 40),
        (2, "Mitte-Ost", 10, 15),
        (3, "Sued", 20, 25),
    ]:
        rows += [
            (f"{nr:03d} {name}", 2030, "alle", "u3", young),
            (f"{nr:03d} {name}", 2030, "alle", "3-5", older),
        ]
    rows += [("Stadt gesamt", 2030, "alle", "u3", 60), ("Stadt gesamt", 2030, "alle", "3-5", 80)]
    pd.DataFrame(rows, columns=["gebiet", "jahr", "sex", "alter", "anzahl"]).to_csv(
        tmp_path / "forecast.csv", index=False
    )
    labels = {
        "pct_welfare_15_64": "SGB2 15-64",
        "pct_welfare_0_14": "SGB2 0-14",
        "pct_children_migration_background": "Migration",
        "pct_children_single_parent": "Alleinerziehend",
    }
    values = {1: [5, 8, 30, 15], 2: [20, 30, 60, 35], 3: [10, 12, 40, 20]}
    social = [
        (f"{nr:03d} X", "31.12.2029", label, str(v[i]))
        for nr, v in values.items()
        for i, label in enumerate(labels.values())
    ]
    pd.DataFrame(social, columns=["gebiet", "datum", "merkmal", "wert"]).to_csv(tmp_path / "social.csv", index=False)
    centres = gpd.GeoSeries([Point(390500 + 1000 * k, 5810500) for k in range(3)], crs=25833).to_crs(3035)
    cells = pd.DataFrame(
        {
            "zelle": ["c1", "c2", "c3"],
            "mx": centres.x.round(),
            "my": centres.y.round(),
            "u3": ["4", "2", "3"],
            "k35": ["6", "1", "2"],
        }
    )
    with zipfile.ZipFile(tmp_path / "census.zip", "w") as archive:
        archive.writestr("grid/zensus_grid.csv", cells.to_csv(sep=";", index=False))
    gpd.GeoDataFrame(
        {"fn": ["1000"], "art": ["ganz"], "bid": ["b1"]},
        geometry=[box(390100, 5810100, 390120, 5810120)],
        crs=25833,
    ).to_file(tmp_path / "buildings.gpkg", layer="bauten", driver="GPKG")
    (tmp_path / "decisions.csv").write_text("building_id,decision\n", encoding="utf-8")
    (tmp_path / "osm.json").write_text("{}", encoding="utf-8")

    data = muenster_data()
    data.pop("validation"), data.pop("reproducibility")
    data["model_parameters"] = {
        "facility_budget": 2,
        "walking_speed_m_per_min": 80.0,
        "least_advantaged_quantile": 0.5,
    }
    data["solver_resources"]["threads"] = 8
    data["city"] = {"name": "Testburg", "scenario": "testburg"}
    data["crs"]["projected"] = 25833
    data["sources"]["districts"] = {
        "file": str(tmp_path / "districts.gpkg"),
        "id_column": "bez_nr",
        "name_column": "bez_name",
        "id_width": 3,
    }
    data["sources"]["forecast"].update(
        file=str(tmp_path / "forecast.csv"),
        year=2030,
        separator=",",
        encoding="utf-8",
        area_column="gebiet",
        year_column="jahr",
        sex_column="sex",
        age_column="alter",
        value_column="anzahl",
        all_sexes_value="alle",
        city_total_marker="gesamt",
        district_code_pattern=r"^(\d{3})\s",
        age_groups={"age_0_2": "u3", "age_3_5": "3-5"},
    )
    data["sources"]["social"].update(
        file=str(tmp_path / "social.csv"),
        year=2029,
        separator=",",
        encoding="utf-8",
        decimal=",",
        area_column="gebiet",
        time_column="datum",
        indicator_column="merkmal",
        value_column="wert",
        district_code_pattern=r"^(\d{3})\s",
        indicators=labels,
    )
    data["sources"]["census"].update(
        file=str(tmp_path / "census.zip"),
        sha256="",
        member_suffix="zensus_grid.csv",
        id_column="zelle",
        x_column="mx",
        y_column="my",
        count_columns=["u3", "k35"],
    )
    data["sources"]["buildings"].update(
        file=str(tmp_path / "buildings.gpkg"),
        sha256="",
        layer="bauten",
        function_column="fn",
        object_type_column="art",
        id_column="bid",
        building_value="ganz",
        part_value="teil",
        mixed_use_code="1100",
        residential_codes={"1000": "Wohnen", "1100": "Gemischt"},
    )
    data["sources"]["dormitories"].update(decisions_file=str(tmp_path / "decisions.csv"), decisions_sha256="")
    data["osm"].update(network_snapshot=str(tmp_path / "osm.json"), snapshot_dir=str(tmp_path))
    data["outputs"] = {k: str(tmp_path / "out" / k) for k in data["outputs"]}
    return write_config(tmp_path, data)


def _run_with_config(path: Path, code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, cfg.CONFIG_ENV: str(path)}
    return subprocess.run(
        [sys.executable, "-c", code], check=False, env=env, capture_output=True, text=True, cwd=cfg.PROJECT_ROOT
    )


def test_a_second_city_runs_through_the_same_loaders(tmp_path):
    config_path = _synthetic_city(tmp_path)
    cfg.check_data(cfg.load_city_config(config_path))  # files present, columns as configured
    code = """
import json
from childcare_access import core, settings
from childcare_access.demand import load_census_children_cells, load_districts, load_forecast_children
from childcare_access.vulnerability import build_vulnerability
districts = load_districts()
children, city = load_forecast_children()
merged, groups = build_vulnerability(districts)
cells = load_census_children_cells(districts)
print(json.dumps({
    "crs": core.PROJECTED_CRS, "district_crs": districts.crs.to_epsg(), "ids": sorted(districts["district_id"]),
    "children": children.set_index("district_id")["children_0_5"].to_dict(), "city": city["children_0_5"],
    "q4": sorted(merged.loc[merged["least_advantaged_district"].astype(bool), "district_name"]),
    "cells": cells.groupby("district_id")["children_0_5"].sum().to_dict(),
    "P": core.P_NEW_KITAS, "speed": core.WALKING_SPEED_M_PER_MIN, "quantile": core.LEAST_ADVANTAGED_QUANTILE,
    "threads": settings.SOLVER_RESOURCES["District Rawlsian (screened)"]["threads"]}))
"""
    result = _run_with_config(config_path, code)
    assert result.returncode == 0, result.stderr[-3000:]
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["crs"] == 25833 and out["district_crs"] == 25833
    assert out["ids"] == ["001", "002", "003"]
    assert out["children"] == {"001": 70, "002": 25, "003": 45} and out["city"] == 140
    assert out["q4"] == ["Mitte-Ost", "Sued"]  # index at or above the configured 50th percentile
    assert out["cells"] == {"001": 10.0, "002": 3.0, "003": 5.0}
    assert (out["P"], out["speed"], out["quantile"], out["threads"]) == (
        2,
        80.0,
        0.5,
        8,
    )  # configured, not Münster's
