"""The section-by-section setup: answers are checked at once and saved, so a mistake needs one section redone."""

from __future__ import annotations

import zipfile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from childcare_access import city_steps as steps


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "PROJECT_ROOT", tmp_path)
    districts = gpd.GeoDataFrame(
        {"code": [1, 2], "name": ["A", "B"]},
        geometry=[box(7.0, 51.0, 7.1, 51.1), box(7.1, 51.0, 7.2, 51.1)],
        crs=4326,
    )
    districts.to_file(tmp_path / "districts.gpkg", driver="GPKG")
    made = steps.Setup("Test City")
    made.set_districts(str(tmp_path / "districts.gpkg"), "code", "name", 25832)
    made.set_demand("building_area")
    return made


def _children(tmp_path, rows):
    path = tmp_path / "children.csv"
    pd.DataFrame(rows, columns=["code", "kids"]).to_csv(path, index=False)
    return str(path)


def test_answers_are_saved_and_come_back(setup):
    again = steps.Setup("Test City")
    assert again.answers["districts"]["epsg"] == 25832 and again.answers["demand"]["mode"] == "building_area"


def test_a_section_cannot_run_before_the_one_it_needs(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "PROJECT_ROOT", tmp_path)
    made = steps.Setup("Other")
    with pytest.raises(steps.StepProblem, match="districts"):
        made.need("districts")


def test_nothing_runs_before_a_city_is_chosen():
    with pytest.raises(steps.StepProblem, match="section 1"):
        steps.Setup().need("districts")


def test_repeated_district_ids_are_refused(tmp_path, setup):
    gpd.GeoDataFrame(
        {"code": [1, 1], "name": ["A", "B"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326
    ).to_file(tmp_path / "bad.gpkg", driver="GPKG")
    with pytest.raises(steps.StepProblem, match="unique"):
        setup.set_districts(str(tmp_path / "bad.gpkg"), "code", "name", 25832)


def test_a_city_wide_total_row_is_reported_by_id(tmp_path, setup):
    with pytest.raises(steps.StepProblem, match="5913000"):
        setup.set_population(
            _children(tmp_path, [(1, 5), (2, 6), (5913000, 11)]),
            "code",
            "kids",
            "",
            "",
            "observed",
            2025,
            "test",
        )


def test_a_district_without_a_row_is_reported(tmp_path, setup):
    with pytest.raises(steps.StepProblem, match="no row"):
        setup.set_population(_children(tmp_path, [(1, 5)]), "code", "kids", "", "", "observed", 2025, "test")


def test_counts_that_are_not_numbers_are_refused(tmp_path, setup):
    with pytest.raises(steps.StepProblem, match="plain numbers"):
        setup.set_population(
            _children(tmp_path, [(1, "1.135,0"), (2, 6)]),
            "code",
            "kids",
            "",
            "",
            "observed",
            2025,
            "test",
        )


def test_a_valid_table_reports_its_total(tmp_path, setup):
    assert "11 children" in setup.set_population(
        _children(tmp_path, [(1, 5), (2, 6)]), "code", "kids", "", "", "observed", 2025, "test"
    )


def test_facilities_outside_the_city_are_refused(tmp_path, setup):
    gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs=4326).to_file(tmp_path / "far.gpkg", driver="GPKG")
    with pytest.raises(steps.StepProblem, match="None of these"):
        setup.set_facilities(str(tmp_path / "far.gpkg"))


def test_points_are_refused_as_buildings(tmp_path, setup):
    setup.set_sites("residential_buildings")
    gpd.GeoDataFrame({"id": [1]}, geometry=[Point(7.05, 51.05)], crs=4326).to_file(
        tmp_path / "points.gpkg", driver="GPKG"
    )
    with pytest.raises(steps.StepProblem, match="footprints"):
        setup.set_buildings(str(tmp_path / "points.gpkg"), "points", "id", "", [], [])


def test_the_priority_summary_names_the_selected_districts(tmp_path, setup):
    path = tmp_path / "index.csv"
    pd.DataFrame({"code": [1, 2], "score": [10, 20]}).to_csv(path, index=False)
    assert "selected: 2" in setup.set_priority(
        "external_index",
        str(path),
        "code",
        value="score",
        name="n",
        definition="d",
        higher=True,
        quantile=0.75,
    )


def test_a_country_zip_with_several_tables_keeps_only_cells_near_the_city(tmp_path, setup):
    setup.set_demand("census_hybrid")
    setup.answers["population"] = {}
    centre = gpd.GeoSeries([Point(7.1, 51.05)], crs=4326).to_crs(3035).iloc[0]
    table = f"ID;x;y;kids\nnear;{centre.x};{centre.y};3\nfar;{centre.x + 500000};{centre.y};9\n"
    path = tmp_path / "country.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("grid_100m.csv", "﻿" + table)
        archive.writestr("grid_1km.csv", table)
    assert setup.census_members(str(path)) == ["grid_100m.csv", "grid_1km.csv"]
    assert setup.census_columns(str(path), "grid_100m.csv") == ["ID", "x", "y", "kids"]
    assert "1 grid cells" in setup.set_census(str(path), "grid_100m.csv", "ID", "x", "y", ["kids"], 3035, 100, 2022)
    with zipfile.ZipFile(setup.prepared / "census_grid.zip") as archive:
        assert archive.namelist() == ["census_grid.csv"] and b"far" not in archive.read("census_grid.csv")


def test_a_grid_far_from_the_city_is_refused(tmp_path, setup):
    setup.set_demand("census_hybrid")
    setup.answers["population"] = {}
    path = tmp_path / "grid.csv"
    path.write_text("ID;x;y;kids\na;1;1;3\n", encoding="utf-8")
    with pytest.raises(steps.StepProblem, match="No grid cells"):
        setup.set_census(str(path), "grid.csv", "ID", "x", "y", ["kids"], 3035, 100, 2022)


def test_the_file_picker_refuses_a_missing_file(tmp_path):
    picker = steps.FilePicker((".csv",), tmp_path)
    with pytest.raises(steps.StepProblem, match="Choose a file"):
        _ = picker.value
    picker.path.value = str(tmp_path / "none.csv")
    with pytest.raises(steps.StepProblem, match="No file"):
        _ = picker.value


def test_the_file_picker_lists_folders_and_matching_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    (tmp_path / "b.pdf").write_text("x", encoding="utf-8")
    picker = steps.FilePicker((".csv",), tmp_path)
    assert [label for label, _ in picker.list.options] == [
        "[up one folder]",
        "[folder] sub",
        "a.csv",
    ]
    picker.list.value = tmp_path / "a.csv"
    assert picker.value == str(tmp_path / "a.csv")


def test_portal_record_numbers_are_not_guessed_as_district_ids():
    columns = ["geographische_koordinaten", "stadtbezirk_nr", "stadtbezirk_bezeichnung", "rowid"]
    assert steps._dropdown("ID", columns, ("id", "nr", "code")).value == "stadtbezirk_nr"
    assert steps._dropdown("Name", columns, ("name", "bezeichnung")).value == "stadtbezirk_bezeichnung"


def test_an_optional_column_without_a_match_stays_empty():
    assert steps._dropdown("Age", ["a", "b"], ("0_5",), optional=True).value == ""


def test_saving_districts_shows_their_ids_and_names(tmp_path, setup):
    assert "IDs look like 1, 2; names look like A, B" in setup.set_districts(
        str(tmp_path / "districts.gpkg"), "code", "name", 25832
    )
