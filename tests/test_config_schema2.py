"""Configuration version 2 (selectable methods), how version-1 files map to methods, and the standard tables."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tomllib

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from childcare_access import requirements as req
from childcare_access import schema
from childcare_access.config import (
    CONFIG_ENV,
    PROJECT_ROOT,
    ConfigError,
    _check_v2,
    check_city,
    load_city_config,
)

TEMPLATE = PROJECT_ROOT / "config" / "city_template.toml"


@pytest.fixture
def v2() -> dict:
    return tomllib.loads(TEMPLATE.read_text(encoding="utf-8"))


def problems(data: dict) -> str:
    return "\n".join(_check_v2(data))


def select(data: dict, **choice) -> dict:
    data = copy.deepcopy(data)
    for dotted, value in choice.items():
        table, key = dotted.rsplit("__", 1)
        node = data
        for part in table.split("__"):
            node = node[part]
        node[key] = value
    return data


# version-1 files
def test_version_1_files_are_read_as_the_default_methods():
    muenster = load_city_config(PROJECT_ROOT / "config" / "muenster.toml")
    assert muenster.version == 1
    assert muenster.methods == {
        "demand": {
            "mode": "census_hybrid",
            "count_type": "projected",
            "reference_year": 2026,
            "census_year": 2022,
        },  # from config/muenster_metadata.toml
        "vulnerability": {"mode": "four_indicator", "quantile": 0.75},
        "candidates": {"source": "residential_buildings"},
        "facilities": {"source": "osm_cache"},
    }
    synthetic = load_city_config(PROJECT_ROOT / "config" / "synthetic_city.toml")
    assert (
        synthetic.methods["demand"]["mode"] == "census_hybrid" and synthetic.methods["facilities"]["source"] == "file"
    )
    assert synthetic.metadata == {} and synthetic.methods["demand"]["census_year"] is None  # no companion file


def test_the_metadata_companion_is_descriptive_only(tmp_path):
    source = PROJECT_ROOT / "config" / "muenster.toml"
    assert set(load_city_config(source).metadata) == {"demand"}
    copy_ = tmp_path / "muenster.toml"
    copy_.write_bytes(source.read_bytes())
    (tmp_path / "muenster_metadata.toml").write_text(
        "[demand]\ncensus_year = 2022\nfacility_budget = 4\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="only descriptive keys"):
        _ = load_city_config(copy_).metadata  # reading the property validates the companion file
    (tmp_path / "muenster_metadata.toml").write_text('[demand]\ncensus_year = "2022"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="wrong type"):
        _ = load_city_config(copy_).metadata
    # nothing but the configuration reads the companion: no other module touches .metadata or the file
    readers = [
        p.name
        for p in (PROJECT_ROOT / "src" / "childcare_access").glob("*.py")
        if p.name != "config.py"
        and (".metadata" in p.read_text(encoding="utf-8") or "_metadata.toml" in p.read_text(encoding="utf-8"))
    ]
    assert readers == []


def test_the_template_is_a_valid_version_2_file_but_not_ready():
    config = load_city_config(TEMPLATE)
    assert config.version == 2
    assert config.methods["demand"] == {
        "mode": "census_hybrid",
        "count_type": "projected",
        "reference_year": 2026,
        "source": "<-- REPLACE: publisher, table, date",
        "census_year": 2022,
    }
    ok, lines = check_city(TEMPLATE)
    text = "\n".join(lines)
    assert not ok and "template placeholders left" in text and "missing dataset [demand.census]" in text
    assert "methods: demand = census_hybrid; vulnerability = four_indicator" in text


def test_the_quantile_comes_from_the_vulnerability_method(v2):
    assert not problems(select(v2, vulnerability__four_indicator__quantile=0.8))
    assert "least_advantaged_quantile is set with the vulnerability method" in problems(
        select(v2, model_parameters__least_advantaged_quantile=0.8)
    )
    assert "strictly between 0 and 1" in problems(select(v2, vulnerability__four_indicator__quantile=1.0))


# ------------------------------------------------------------------------------------------------ requirements
def test_a_missing_method_table_is_named_with_the_explicit_alternative_never_switched(v2):
    no_census = copy.deepcopy(v2)
    del no_census["demand"]["census"]
    text = problems(no_census)
    assert "missing table [demand.census] (needed by demand.mode = 'census_hybrid')" in text
    assert "select demand.mode = 'building_area' or 'direct' explicitly" in text
    assert not problems(select(no_census, demand__mode="building_area"))  # only when chosen explicitly

    no_index = copy.deepcopy(v2)
    del no_index["vulnerability"]["four_indicator"]
    assert "select vulnerability.mode = 'external_index' or 'priority_districts' explicitly" in problems(no_index)
    no_population = copy.deepcopy(v2)
    del no_population["population"]
    assert "missing table [population]" in problems(select(no_population, demand__mode="building_area"))


def test_unknown_method_names_are_rejected(v2):
    for key, value in [
        ("demand__mode", "census"),
        ("vulnerability__mode", "q4"),
        ("candidates__source", "all"),
        ("facilities__source", "web"),
    ]:
        assert "must be one of" in problems(select(v2, **{key: value}))


def test_child_counts_need_a_count_type_year_source_and_consistent_age_columns(v2):
    assert "population.count_type] must be one of ['observed', 'projected']" in problems(
        select(v2, population__count_type="forecast")
    )
    assert "give both age_0_2 and age_3_5 or neither" in problems(select(v2, population__age_0_2="u3"))
    assert "give children_0_5, or age_0_2 and age_3_5" in problems(select(v2, population__children_0_5=""))
    assert not problems(select(v2, population__children_0_5="", population__age_0_2="u3", population__age_3_5="k35"))
    assert "source must not be empty" in problems(select(v2, population__source=" "))
    no_year = copy.deepcopy(v2)
    del no_year["population"]["reference_year"]
    assert "missing key [population] reference_year" in problems(no_year)


def test_direct_demand_points_and_polygons(v2):
    points = select(v2, demand__mode="direct", candidates__source="file", candidates__file="sites.gpkg")
    del points["buildings"], points["population"], points["demand"]["census"]
    assert not problems(points)  # no buildings, totals or grid needed
    assert "candidates.source = 'residential_buildings'" in problems(
        select(points, candidates__source="residential_buildings")
    )
    polygons = select(points, demand__direct__geometry="polygons")
    assert "polygon_allocation = 'building_area'" in problems(polygons)  # needs the buildings
    assert not problems(select(polygons, demand__direct__polygon_allocation="network_nodes"))
    assert "polygon_allocation] must be one of" in problems(
        select(polygons, demand__direct__polygon_allocation="centroid")
    )
    assert "boundary_rule] must be one of ['reject', 'split']" in problems(
        select(
            polygons,
            demand__direct__polygon_allocation="network_nodes",
            demand__direct__boundary_rule="centroid",
        )
    )


def test_external_index_and_priority_list_rules(v2):
    external = select(v2, vulnerability__mode="external_index")
    assert not problems(external)
    del external["vulnerability"]["external_index"]["higher_is_more_vulnerable"]
    assert "missing key [vulnerability.external_index] higher_is_more_vulnerable" in problems(external)
    assert "ties] must be one of ['include']" in problems(
        select(v2, vulnerability__mode="external_index", vulnerability__external_index__ties="exclude")
    )
    assert "missing] must be one of ['error', 'not_priority']" in problems(
        select(v2, vulnerability__mode="external_index", vulnerability__external_index__missing="drop")
    )
    assert "expected_count must be an integer >= 0" in problems(
        select(
            v2,
            vulnerability__mode="external_index",
            vulnerability__external_index__expected_count=-1,
        )
    )
    priority = select(v2, vulnerability__mode="priority_districts")
    assert not problems(priority)
    assert "label must not be empty" in problems(select(priority, vulnerability__priority_districts__label=""))


def test_candidate_rules(v2):
    assert "candidates.shared_node] must be one of ['error', 'keep_first']" in problems(
        select(v2, candidates__shared_node="any")
    )
    assert "max_snap_m must be > 0" in problems(select(v2, candidates__max_snap_m=0))
    supplied = select(v2, candidates__source="file")
    del supplied["candidates"]["file"]
    assert "missing key [candidates] file (needed by candidates.source = 'file')" in problems(supplied)


def test_input_files_follow_the_selected_methods(v2):
    assert [t for t, _ in req.input_files(v2)] == [
        "sources.districts",
        "osm",
        "population",
        "demand.census",
        "buildings",
        "vulnerability.four_indicator",
        "facilities",
    ]
    direct = select(
        v2,
        demand__mode="direct",
        candidates__source="file",
        vulnerability__mode="priority_districts",
    )
    del direct["population"]
    assert [t for t, _ in req.input_files(direct)] == [
        "sources.districts",
        "osm",
        "demand.direct",
        "vulnerability.priority_districts",
        "facilities",
        "candidates",
    ]


def test_a_version_2_city_without_unused_tables_imports_and_builds_until_its_data(v2, tmp_path):
    text = TEMPLATE.read_text(encoding="utf-8")
    for start, end in [
        ("[population]", "[demand.direct]"),
        ("[buildings]", "# [buildings.dormitories]"),
        ("[vulnerability.four_indicator]", "[vulnerability.external_index]"),
    ]:
        text = text[: text.index(start)] + text[text.index(end) :]
    text = text.replace('mode = "census_hybrid"', 'mode = "direct"').replace(
        'mode = "four_indicator"', 'mode = "priority_districts"'
    )
    text = text.replace('source = "residential_buildings"', 'source = "file"')
    path = tmp_path / "city.toml"
    path.write_text(text, encoding="utf-8")
    assert load_city_config(path).methods["demand"]["mode"] == "direct"
    code = (
        "import childcare_access.settings as s, childcare_access.demand, childcare_access.vulnerability, "
        "childcare_access.validation, childcare_access.network, childcare_access.optimisation, "
        "childcare_access.workflow as w\n"
        "print(s.FORECAST_FILE, s.CENSUS_ZIP, s.ALKIS_GPKG, s.SOCIAL_FILE, s.STUDENT_DECISIONS_FILE, s.KITA_FILE)\n"
        f"try:\n    w.build(r'{tmp_path / 'workspace'}')\nexcept FileNotFoundError as e:\n    print('missing:', e)"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={**os.environ, CONFIG_ENV: str(path)},
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert "None None None None None data/example_city/facilities.geojson" in run.stdout
    assert "missing: [sources.districts.file]" in run.stdout  # every method is implemented: only its data are absent


def test_a_broken_version_2_file_lists_every_problem(tmp_path):
    path = tmp_path / "city.toml"
    path.write_text("[schema]\nversion = 2\n[city]\nname = 'x'\n", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_city_config(path)
    assert "missing table [osm]" in str(error.value) and "missing key [city] scenario" in str(error.value)
    path.write_text("[schema]\nversion = 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="version must be 1 or 2"):
        load_city_config(path)


# ------------------------------------------------------------------------------------------------ schema
def districts() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"district_id": ["1", "2"], "district_name": ["A", "B"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=25832,
    )


def test_valid_standard_tables_pass():
    d = districts()
    schema.check("districts", d)
    schema.check("child_totals", pd.DataFrame({"district_id": ["1", "2"], "children_0_5": [3.0, 4.0]}), d)
    schema.check(
        "priority_groups",
        pd.DataFrame(
            {
                "district_id": ["1", "2"],
                "district_name": ["A", "B"],
                "priority": [True, False],
                "group": ["priority", "other"],
            }
        ),
        d,
    )
    origins = pd.DataFrame(
        {
            "origin_id": [0, 1],
            "district_id": ["1", "2"],
            "x": [0.5, 1.5],
            "y": [0.5, 0.5],
            "children_0_5": [3.0, 4.0],
            "age_0_2": [1.0, 2.0],
            "age_3_5": [2.0, 2.0],
            "origin_type": ["supplied_point", "building"],
            "source_id": ["p1", "b7"],
        }
    )
    schema.check("demand_origins", origins, d)
    totals = pd.DataFrame({"district_id": ["1", "2"], "children_0_5": [3.0, 4.0]})
    assert (schema.conservation(origins, totals).abs() <= schema.TOLERANCE).all()


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda f: f.drop(columns="x"), "missing column x"),
        (lambda f: f.assign(origin_id=[0, 0]), "origin_id: duplicates"),
        (lambda f: f.assign(children_0_5=[-1.0, 4.0]), "negative counts"),
        (lambda f: f.assign(children_0_5=[None, 4.0]), "missing values"),
        (lambda f: f.assign(age_0_2=[1.0, 2.0]), "only age_0_2 given"),
        (lambda f: f.assign(age_0_2=[1.0, 2.0], age_3_5=[1.0, 2.0]), "differs from children_0_5"),
        (
            lambda f: f.assign(origin_type=["centroid", "building"]),
            "origin_type: unknown ['centroid']",
        ),
        (lambda f: f.assign(district_id=["1", "9"]), "district_id not in the districts: ['9']"),
    ],
)
def test_invalid_demand_origins_are_rejected(change, message):
    origins = pd.DataFrame(
        {
            "origin_id": [0, 1],
            "district_id": ["1", "2"],
            "x": [0.5, 1.5],
            "y": [0.5, 0.5],
            "children_0_5": [3.0, 4.0],
            "origin_type": ["supplied_point", "building"],
            "source_id": ["p1", "b7"],
        }
    )
    with pytest.raises(schema.SchemaError, match=message.replace("[", r"\[").replace("]", r"\]")):
        schema.check("demand_origins", change(origins), districts())


def test_district_coverage_and_crs():
    with pytest.raises(schema.SchemaError, match="districts without a row"):
        schema.check("child_totals", pd.DataFrame({"district_id": ["1"], "children_0_5": [3.0]}), districts())
    with pytest.raises(schema.SchemaError, match="not in a projected CRS"):
        schema.check("districts", districts().to_crs(4326))
    with pytest.raises(schema.SchemaError, match="priority: must be true/false"):
        schema.check(
            "priority_groups",
            pd.DataFrame(
                {
                    "district_id": ["1", "2"],
                    "district_name": ["A", "B"],
                    "priority": ["yes", "no"],
                    "group": ["p", "o"],
                }
            ),
        )
