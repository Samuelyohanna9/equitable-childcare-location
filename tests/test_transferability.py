"""Transferability: the city template, the configuration check and the data adapters."""

from __future__ import annotations

import zipfile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from childcare_access import adapters
from childcare_access.config import PROJECT_ROOT, check_city, load_city_config

TEMPLATE = PROJECT_ROOT / "config" / "city_template_schema1.toml"


def test_the_template_is_valid_but_not_ready():
    config = load_city_config(TEMPLATE)
    assert set(config.get("sources.social.indicators")) == set(adapters.INDICATOR_KEYS)
    ok, lines = check_city(TEMPLATE)
    text = "\n".join(lines)
    assert not ok and "template placeholders left" in text and "missing dataset [sources.forecast.file]" in text


def test_forecast_and_indicator_tables_are_reshaped_not_changed():
    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["A", "B"],
            "u3": [10, 20],
            "k35": [11, 21],
            "w64": [5.0, 7.0],
            "w14": [8.0, 9.0],
            "mig": [30.0, 40.0],
            "sp": [12.0, 15.0],
        }
    )
    forecast = adapters.forecast_long(wide, "id", {"age_0_2": "u3", "age_3_5": "k35"}, year=2030, name_column="name")
    assert forecast.loc[forecast["area"] == "1 A", "value"].tolist() == [10, 11]
    assert forecast.loc[forecast["area"] == "City total", "value"].tolist() == [30, 32]
    labels = {
        "pct_welfare_15_64": "w64",
        "pct_welfare_0_14": "w14",
        "pct_children_migration_background": "mig",
        "pct_children_single_parent": "sp",
    }
    social = adapters.indicators_long(wide, "id", labels, date="31.12.2029", year=2029)
    with pytest.raises(ValueError, match="must end with the indicator year 2029"):
        adapters.indicators_long(wide, "id", labels, date="2029-12-31", year=2029)
    assert len(social) == 8 and social.set_index(["area", "indicator"]).at[("2", "mig"), "value"] == 40.0
    with pytest.raises(
        ValueError,
        match="must map all four index indicators; missing \\['pct_children_single_parent'\\]",
    ):
        adapters.indicators_long(
            wide,
            "id",
            {k: v for k, v in labels.items() if k != "pct_children_single_parent"},
            date="2029",
        )
    with pytest.raises(ValueError, match="must map"):
        adapters.forecast_long(wide, "id", {"age_0_2": "u3"}, year=2030)


def test_buildings_from_any_footprint_source():
    source = gpd.GeoDataFrame(
        {"ref": ["a", "b", "c"], "use": ["house", "shop above flats", "office"]},
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)],
        crs=25832,
    )
    layer = adapters.buildings_layer(
        source,
        residential=source["use"].ne("office"),
        mixed_use=source["use"].str.contains("shop"),
        id_column="ref",
    )
    assert layer["function"].tolist() == ["residential", "mixed", "other"] and layer["building_id"].tolist() == [
        "a",
        "b",
        "c",
    ]
    with pytest.raises(ValueError, match="unique"):
        adapters.buildings_layer(source.assign(ref="x"), residential=True, id_column="ref")
    lines = gpd.GeoDataFrame({"ref": ["l"]}, geometry=[LineString([(0, 0), (1, 1)])], crs=25832)
    with pytest.raises(ValueError, match="not polygons"):
        adapters.buildings_layer(lines, residential=True, id_column="ref")


def test_census_grid_and_facility_files(tmp_path):
    cells = pd.DataFrame({"x": [4157650.0], "y": [3208750.0], "a": [3], "b": [4]})
    path = adapters.census_grid_zip(
        cells, tmp_path / "grid.zip", 100, "x", "y", {"children_0_2": "a", "children_3_5": "b"}
    )
    with zipfile.ZipFile(path) as archive:
        text = archive.read("census/grid.csv").decode()
    assert text.splitlines()[0] == "cell_id;x_centre;y_centre;children_0_2;children_3_5" and "cell_41576_32087" in text
    out = adapters.facilities_file(
        gpd.GeoDataFrame({"n": ["K"]}, geometry=[Point(400005, 5750005)], crs=25832),
        tmp_path / "f.geojson",
        "n",
    )
    assert gpd.read_file(out).crs.to_epsg() == 4326
