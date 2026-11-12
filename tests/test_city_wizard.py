"""The question-by-question city setup: answers are validated, and it writes settings the package accepts."""

from __future__ import annotations

import tomllib

import geopandas as gpd
import pandas as pd
import pytest

from childcare_access import adapters
from childcare_access import city_wizard as wizard


def _questions(*answers):
    replies = iter(answers)
    return wizard.Questions(lambda prompt: next(replies))


def test_a_column_can_be_picked_by_number_or_name_and_bad_answers_are_asked_again():
    assert _questions("", "x", "9", "2").column("Pick", ["a", "b"]) == "b"
    assert _questions("a").column("Pick", ["a", "b"]) == "a"


def test_an_optional_column_can_be_skipped():
    assert _questions("").column("Pick", ["a"], optional=True) == ""


def test_q_stops_the_setup():
    with pytest.raises(wizard.WizardStopped):
        _questions("q").text("Name")


def test_numbers_are_checked_against_their_range():
    assert _questions("two", "1.5", "0.75").number("Share", kind=float, low=0, high=1) == 0.75


def test_a_missing_file_is_asked_again(tmp_path):
    file = tmp_path / "x.csv"
    file.write_text("a\n1\n", encoding="utf-8")
    assert _questions("no/such/file.csv", str(file)).file("Path") == file


def test_the_settings_are_valid_toml_with_every_value_type():
    data = {
        "schema": {"version": 2},
        "demand": {
            "mode": "direct",
            "census": {"cell": 100.0, "count_columns": ["a", "b"], "flag": True},
        },
        "vulnerability": {"pattern": r"^(\S+)"},
        "path": {"file": "C:/data/ü.csv"},
    }
    assert tomllib.loads(wizard.to_toml(data)) == data


def test_a_semicolon_table_is_read_with_its_separator(tmp_path):
    file = tmp_path / "t.csv"
    file.write_text("id;children\n1;5\n", encoding="utf-8")
    frame, separator, encoding = wizard._read_table(file)
    assert list(frame.columns) == ["id", "children"] and separator == ";" and encoding == "utf-8"
    assert isinstance(frame, pd.DataFrame)


def test_b_goes_back_and_the_earlier_answers_are_replayed(capsys):
    q = _questions("Testburg", "2", "b")
    assert q.text("City") == "Testburg" and q.choice("Pick", {"a": "A", "b": "B"}) == "b"
    with pytest.raises(wizard.GoBack):
        q.text("Next")
    again = wizard.Questions(lambda prompt: "Other", replay=q.log[:-1])
    assert again.text("City") == "Testburg" and again.text("Next") == "Other"
    assert "City: Testburg" in capsys.readouterr().out  # a replayed answer is shown, not asked


def test_a_rejected_replayed_answer_stops_the_replay():
    q = wizard.Questions(lambda prompt: "1", replay=["9", "later"])
    assert q.choice("Pick", {"a": "A"}) == "a" and q.replay == [] and q.log == ["1"]


def test_confirm_waits_for_y_and_b_goes_back():
    _questions("x", "y").confirm("Right?")
    with pytest.raises(wizard.GoBack):
        _questions("b").confirm("Right?")


def test_columns_are_shown_with_example_values(capsys):
    _questions("1").column("Pick", ["id", "name"], examples={"id": "1, 2", "name": "A, B"})
    out = capsys.readouterr().out
    assert "id    e.g. 1, 2" in out and "name  e.g. A, B" in out


def test_repeated_column_names_are_numbered():
    assert adapters.unique_names(["a", "a", "b", "a"]) == ["a", "a_2", "b", "a_3"]
    assert adapters.unique_names(["a", "a", "a_2"]) == ["a", "a_3", "a_2"]


def test_a_map_file_with_repeated_column_names_is_copied_with_unique_names(tmp_path):
    shapes = tmp_path / "districts.csv"  # GDAL keeps repeated names here, as in some shapefiles
    shapes.write_text(
        "WKT,Bezirk,Bezirk,rowid\n"
        '"POLYGON((0 0,1 0,1 1,0 1,0 0))",Nord,Nord,1\n'
        '"POLYGON((1 0,2 0,2 1,1 1,1 0))",South,South,2\n',
        encoding="utf-8",
    )
    used, names = adapters.map_with_unique_columns(shapes, tmp_path / "prepared", "districts")
    assert names == ["WKT", "Bezirk", "Bezirk_2", "rowid"] and used.suffix == ".gpkg"
    assert gpd.read_file(used)["Bezirk_2"].tolist() == ["Nord", "South"]
    assert adapters.map_with_unique_columns(used, tmp_path, "x") == (used, names)  # already unique: used as it is


def test_the_network_can_be_downloaded_reused_or_explained(tmp_path, monkeypatch):
    import json

    from childcare_access import osm_download

    districts = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(["POLYGON((0 0,1 0,1 1,0 1,0 0))"]), crs=4326)
    folder, calls = tmp_path / "data" / "x" / "prepared", []

    def fake_download(_districts, target, say=print):
        calls.append(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"osm3s": {"timestamp_osm_base": "2026-01-01T00:00:00Z"}}), encoding="utf-8")
        return {"date": "2026-01-01T00:00:00Z"}

    monkeypatch.setattr(osm_download, "download_network", fake_download)
    for _ in range(2):  # the second time the file is there and is used again (going back does not download again)
        config = {"osm": {}}
        wizard._network(_questions("2"), config, districts, folder)
        assert config["osm"]["snapshot_timestamp"] == "2026-01-01T00:00:00Z"
        assert config["osm"]["network_snapshot"].endswith("data/x/osm/highways.json")
    assert len(calls) == 1
    with pytest.raises(wizard.WizardStopped, match="overpass-turbo"):
        wizard._network(_questions("3"), {"osm": {}}, districts, folder)

    def failing(*_args, **_kwargs):
        raise osm_download.DownloadFailed("every server is busy")

    monkeypatch.setattr(osm_download, "download_network", failing)
    with pytest.raises(wizard.WizardStopped, match="continues where it stopped"):
        wizard._network(_questions("2"), {"osm": {}}, districts, tmp_path / "other" / "prepared")


def test_a_stop_message_is_shown_without_a_traceback(capsys):
    from childcare_access.guided_run import GuidedRunError

    error = GuidedRunError("Do this next.")
    assert error._render_traceback_() == error._render_traceback_()  # asked twice, printed once
    assert capsys.readouterr().out == "Do this next.\n"


def _grid_zip(path, rows):
    """A small zip shaped like the German Zensus grid: the 100 m table and another CSV."""
    import zipfile

    header = "GITTER_ID_100M;x_mp_100m;y_mp_100m;Unter3;a3bis5\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("grid/Zensus2022_Alter_INFR_100m-Gitter.csv", "﻿" + header + "".join(rows))
        archive.writestr("grid/Zensus2022_Alter_INFR_1km-Gitter.csv", header)
    return path


def _district_3035(x0, y0, x1, y1):
    from shapely.geometry import box

    return gpd.GeoDataFrame(geometry=[box(x0, y0, x1, y1)], crs=3035)


def test_the_zensus_grid_is_cut_to_the_cells_near_the_city_and_reused(tmp_path):
    rows = ["a;4100050;3100050;3;–\n", "b;4100950;3100050;1;2\n", "far;4300050;3300050;5;5\n"]
    source = _grid_zip(tmp_path / "zensus.zip", rows)
    member = "grid/Zensus2022_Alter_INFR_100m-Gitter.csv"
    target = tmp_path / "prepared" / "census_grid.zip"
    districts = _district_3035(4100000, 3100000, 4100500, 3100500)
    assert wizard.keep_cells_near(source, member, districts, wizard.ZENSUS, target) == 2  # within 1 km, not "far"
    cut = pd.read_csv(target, sep=";", dtype=str)
    assert cut["GITTER_ID_100M"].tolist() == ["a", "b"] and cut["a3bis5"].tolist() == ["–", "2"]  # values unchanged
    source.unlink()  # the same area again: the earlier result is used, the source is not read
    assert wizard.keep_cells_near(source, member, districts, wizard.ZENSUS, target) == 2


def test_the_zensus_choice_fills_in_the_grid_settings(tmp_path, monkeypatch):
    source = _grid_zip(tmp_path / "zensus.zip", ["a;4100050;3100050;3;1\n"])
    monkeypatch.setattr(wizard, "download_zensus", lambda: source)
    config = {"demand": {"census": {}}}
    wizard._census(_questions("2"), config, _district_3035(4100000, 3100000, 4100500, 3100500), tmp_path / "p")
    grid = config["demand"]["census"]
    assert grid["count_columns"] == ["Unter3", "a3bis5"] and grid["crs"] == 3035 and grid["reference_year"] == 2022
    assert grid["member_suffix"] == "census_grid.csv" and grid["file"].endswith("p/census_grid.zip")
    with pytest.raises(wizard.WizardStopped, match="Germany only"):  # districts far from every cell
        wizard._census(_questions("2"), {"demand": {"census": {}}}, _district_3035(0, 0, 1, 1), tmp_path / "q")


def test_a_zip_with_several_tables_asks_which_one(tmp_path):
    source = _grid_zip(tmp_path / "zensus.zip", ["a;4100050;3100050;3;1\n"])
    config = {"demand": {"census": {}}}
    answers = ["1", str(source), "1", "1", "2", "3", "4,5", "", "", "2022"]  # the 100 m table is the first CSV
    wizard._census(_questions(*answers), config, _district_3035(0, 0, 1, 1), tmp_path / "p")
    grid = config["demand"]["census"]
    assert grid["member_suffix"] == "Zensus2022_Alter_INFR_100m-Gitter.csv" and grid["id_column"] == "GITTER_ID_100M"


def test_values_can_be_picked_by_numbers_and_ranges():
    kinds = ["a", "b", "c", "d", "e"]
    assert _questions("1,3-4").values("Pick", kinds) == ["a", "c", "d"]
    assert _questions("9", "2-2").values("Pick", kinds) == ["b"]


def test_a_word_searches_the_values_and_all_with_takes_every_match(capsys):
    kinds = ["Garage", "Wohngebaeude", "Wohn- und Geschaeftsgebaeude", "Kirche"]
    q = _questions("wohn", "all with wohn")
    assert q.values("Pick", kinds, counts={"Garage": 9, "Wohngebaeude": 5}, unit="buildings") == kinds[1:3]
    out = capsys.readouterr().out
    assert "2 value(s) contain 'wohn'" in out and "(9 buildings)" in out
    assert q.log == ["all with wohn"]  # the search itself is not kept as an answer


def test_a_long_list_shows_the_most_common_values_first(capsys):
    kinds = [f"use {i:03d}" for i in range(100)]
    _questions("1").values("Pick", kinds, counts={"use 077": 50})
    out = capsys.readouterr().out
    assert out.index("use 077") < out.index("use 000") and "... and 60 more" in out


def test_a_file_can_be_looked_at_before_going_on(tmp_path, capsys):
    file = tmp_path / "t.csv"
    file.write_text("id;children\n1;5\n", encoding="utf-8")
    replies = iter([str(file), "2", ""])
    q = wizard.Questions(lambda prompt: next(replies), preview=True)
    assert q.file("Path") == file and q.log == [str(file)]
    assert "The first 1 rows of t.csv" in capsys.readouterr().out
    replies = iter([str(file), "b"])
    with pytest.raises(wizard.GoBack):  # going back from here asks for the file again
        wizard.Questions(lambda prompt: next(replies), preview=True).file("Path")


def test_a_map_file_is_shown_as_a_table_and_on_a_map(tmp_path, capsys, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    file = tmp_path / "d.gpkg"
    gpd.GeoDataFrame({"id": ["1", "2"]}, geometry=gpd.points_from_xy([0, 1], [0, 1]), crs=25832).to_file(file)
    wizard.show_file(file, "table")
    wizard.show_file(file, "map", outline=gpd.read_file(file).buffer(5).to_frame("geometry"))
    out = capsys.readouterr().out
    assert "The first 2 of 2 rows of d.gpkg" in out and "Drawing 2 shapes" in out


def _two_districts():
    return gpd.GeoDataFrame(
        {"oid": [1532625, 1532626], "code": ["1", "2"]}, geometry=gpd.points_from_xy([0, 1], [0, 1]), crs=25832
    )


def test_district_ids_of_a_table_are_checked_and_the_district_column_can_be_changed(tmp_path, capsys):
    table = tmp_path / "children.csv"
    table.write_text("district;children\n1;5\n2;7\n", encoding="utf-8")
    config = {"sources": {"districts": {"id_column": "oid"}}}
    frame = wizard._read_table(table, 20)[0]
    q = _questions("1", "2", "2")  # the table's district column; "pick another ID column of the district file"; code
    assert wizard._district_id_column(q, config, _two_districts(), table, frame) == "district"
    assert config["sources"]["districts"]["id_column"] == "code"
    out = capsys.readouterr().out
    assert "These IDs do not match the district IDs" in out and "e.g. 1532625, 1532626" in out


def test_matching_district_ids_ask_nothing_more(tmp_path):
    table = tmp_path / "children.csv"
    table.write_text("district,children\n1.0,5\n2,7\n", encoding="utf-8")
    config = {"sources": {"districts": {"id_column": "code"}}}
    q = _questions("1")
    assert (
        wizard._district_id_column(q, config, _two_districts(), table, wizard._read_table(table, 20)[0]) == "district"
    )
    assert q.log == ["1"]


def test_leading_zeros_are_named_as_the_difference():
    assert "leading zeros" in wizard._id_problem({"01", "02"}, {"1", "2"}, every=True)
    assert wizard._id_problem({"1"}, {"1", "2"}, every=False) == ""


def test_a_facility_id_column_with_repeated_values_is_asked_again(tmp_path, capsys):
    file = tmp_path / "kitas.gpkg"
    gpd.GeoDataFrame(
        {"area": ["1", "1"], "name": ["A", "B"]}, geometry=gpd.points_from_xy([0, 1], [0, 1]), crs=25832
    ).to_file(file)
    config = {"facilities": {}}
    q = _questions(str(file), "1", "2")
    wizard._facilities(q, config, tmp_path)
    assert config["facilities"]["id_column"] == "name" and q.log == [str(file), "2"]
    assert "area is not unique" in capsys.readouterr().out
