"""Small converters from common source layouts to the input formats a city configuration describes.

The loaders read long tables, a building layer with a use code, a zipped census grid CSV and a facility file (see
``config/city_template_schema1.toml``). These helpers write such files from typical alternatives, e.g. a wide table with
one row per district, or building footprints from any provider. They only reshape and rename; they never estimate,
impute or substitute values, so the meaning of every variable stays that of the source. Column and code names follow the
template defaults; point the configuration at the written files.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import pandas as pd

AGE_KEYS = ("age_0_2", "age_3_5")
INDICATOR_KEYS = (
    "pct_welfare_15_64",
    "pct_welfare_0_14",
    "pct_children_migration_background",
    "pct_children_single_parent",
)


def _area_labels(frame: pd.DataFrame, id_column: str, name_column: str | None) -> pd.Series:
    """Area labels: the ID, followed by the name if there is a name column."""
    ids = frame[id_column].astype(str)
    return ids + " " + frame[name_column].astype(str) if name_column else ids


def forecast_long(
    frame: pd.DataFrame,
    id_column: str,
    age_columns: dict[str, str],
    year: int,
    name_column: str | None = None,
    city_total: bool = True,
) -> pd.DataFrame:
    r"""Wide forecast (one row per district, one column per age group) -> long table of the template layout.

    ``age_columns`` maps the method's keys ``age_0_2`` and ``age_3_5`` to the source columns holding children of
    exactly those ages. Area labels are "<id> <name>" (configure ``district_code_pattern = '^(\d+)'`` for numeric
    IDs).
    """
    missing = [k for k in AGE_KEYS if k not in age_columns]
    if missing:
        raise ValueError(f"age_columns must map {list(AGE_KEYS)}; missing {missing}")
    labels = {"age_0_2": "0-2", "age_3_5": "3-5"}
    area = _area_labels(frame, id_column, name_column)
    rows = [
        {"area": a, "year": year, "sex": "total", "age_group": labels[key], "value": v}
        for key in AGE_KEYS
        for a, v in zip(area, frame[age_columns[key]], strict=True)
    ]
    if city_total:
        rows += [
            {
                "area": "City total",
                "year": year,
                "sex": "total",
                "age_group": labels[key],
                "value": frame[age_columns[key]].sum(),
            }
            for key in AGE_KEYS
        ]
    out = pd.DataFrame(rows, columns=["area", "year", "sex", "age_group", "value"])
    if out["value"].isna().any() or (out["value"] < 0).any():
        raise ValueError("forecast values must be complete and non-negative")
    return out


def indicators_long(
    frame: pd.DataFrame,
    id_column: str,
    indicator_columns: dict[str, str],
    date: str,
    name_column: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """Wide indicator table (one row per district) -> long table of the template layout.

    ``indicator_columns`` maps each of the four index indicators to the source column measuring that concept as a
    percentage per district. The output indicator label is the source column name; configure
    ``[sources.social.indicators]`` with these labels and document each in ``indicator_definitions``. The loader
    selects rows whose ``date`` ENDS with the configured year (e.g. "31.12.2029" or "2029"); pass ``year`` to check it.
    """
    if year is not None and not str(date).endswith(str(year)):
        raise ValueError(
            f"date {date!r} must end with the indicator year {year} (e.g. '31.12.{year}'), as the loader selects by it"
        )

    missing = [k for k in INDICATOR_KEYS if k not in indicator_columns]
    if missing:
        raise ValueError(
            f"indicator_columns must map all four index indicators; missing {missing} "
            "(the index needs all four; an equivalent indicator must measure the same concept and be documented)"
        )
    area = _area_labels(frame, id_column, name_column)
    rows = [
        {"area": a, "date": date, "indicator": column, "value": v}
        for column in indicator_columns.values()
        for a, v in zip(area, frame[column], strict=True)
    ]
    out = pd.DataFrame(rows, columns=["area", "date", "indicator", "value"])
    if out["value"].isna().any():
        raise ValueError("indicator values must be complete for every district")
    return out


def buildings_layer(
    footprints: gpd.GeoDataFrame,
    residential: Iterable[str],
    mixed_use: Iterable[str] | None = None,
    id_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Building footprints from any source -> layer with ``function``, ``object_type`` and ``building_id``.

    ``function`` is residential, mixed or other; ``object_type`` is "building". ``residential`` and ``mixed_use`` are
    boolean masks the user derives from the source's own use classification; non-residential buildings are kept as
    "other" (they receive no demand).
    """
    if footprints.crs is None:
        raise ValueError("footprints have no CRS")
    residential = pd.Series(residential, index=footprints.index).astype(bool)
    mixed = pd.Series(mixed_use if mixed_use is not None else False, index=footprints.index).astype(bool)
    function = pd.Series("other", index=footprints.index)
    function[residential] = "residential"
    function[residential & mixed] = "mixed"
    ids = (
        footprints[id_column].astype(str)
        if id_column
        else pd.Series([f"b{i}" for i in range(len(footprints))], index=footprints.index)
    )
    if ids.duplicated().any():
        raise ValueError("building IDs must be unique")
    polygons = footprints.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if not polygons.all():
        raise ValueError(f"{int((~polygons).sum())} footprints are not polygons")
    return gpd.GeoDataFrame(
        {"function": function, "object_type": "building", "building_id": ids},
        geometry=footprints.geometry,
        crs=footprints.crs,
    )


def census_grid_zip(
    cells: pd.DataFrame,
    path: str | Path,
    cell_size_m: float,
    x_column: str,
    y_column: str,
    count_columns: dict[str, str],
    id_column: str | None = None,
    member: str = "census/grid.csv",
) -> Path:
    """Gridded census counts -> zip with one ``;``-separated CSV of children per cell.

    The CSV has ``cell_id``, ``x_centre``, ``y_centre``, ``children_0_2`` and ``children_3_5``. ``x_column`` and
    ``y_column`` must be cell-CENTRE coordinates in the grid CRS; ``count_columns`` maps ``children_0_2`` and
    ``children_3_5`` to the source columns.
    """
    missing = [k for k in ("children_0_2", "children_3_5") if k not in count_columns]
    if missing:
        raise ValueError(f"count_columns must map {missing}")
    out = pd.DataFrame(
        {
            "cell_id": cells[id_column].astype(str)
            if id_column
            else [
                f"cell_{int(x // cell_size_m)}_{int(y // cell_size_m)}"
                for x, y in zip(cells[x_column], cells[y_column], strict=True)
            ],
            "x_centre": cells[x_column],
            "y_centre": cells[y_column],
            **{k: cells[v] for k, v in count_columns.items()},
        }
    )
    path = Path(path)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, out.to_csv(sep=";", index=False))
    return path


def unique_names(names: Iterable[str]) -> list[str]:
    """Column names with repeats numbered, in the same order: ``["a", "a", "b"]`` -> ``["a", "a_2", "b"]``.

    Some files (for example shapefiles, whose names are cut to 10 characters) hold two columns with the same name;
    a column could then not be chosen by its name. Only the later repeats are renamed.
    """
    names = [str(n) for n in names]
    result: list[str] = []
    for name in names:
        new, number = name, 1
        while new in result or (new != name and new in names):
            number += 1
            new = f"{name}_{number}"
        result.append(new)
    return result


def map_with_unique_columns(path: str | Path, folder: str | Path, stem: str) -> tuple[Path, list[str]]:
    """A map file whose column names are all different, and those names (the geometry column excluded).

    If the file repeats a column name, a copy with numbered names (see :func:`unique_names`) is written to
    ``<folder>/<stem>.gpkg`` and returned; otherwise the file itself is returned. Values are copied unchanged.
    """
    import pyogrio

    path = Path(path)
    fields = [str(f) for f in pyogrio.read_info(path)["fields"]]
    names = unique_names(fields)
    if names == fields:
        return path, names
    frame = gpd.read_file(path)
    columns = list(frame.columns)
    geometry = frame.geometry.name
    attributes = [c for c in columns if c != geometry]
    if len(attributes) != len(names):
        raise ValueError(f"{path.name}: could not match its {len(fields)} columns")
    frame.columns = [names.pop(0) if c != geometry else c for c in columns]
    copy = Path(folder) / f"{stem}.gpkg"
    copy.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(copy, layer=stem, driver="GPKG")
    return copy, [c for c in frame.columns if c != geometry]


def facilities_file(facilities: gpd.GeoDataFrame, path: str | Path, name_column: str | None = None) -> Path:
    """Existing facilities (points or polygons, any CRS) -> GeoJSON for ``osm.kita_file``."""
    if facilities.crs is None:
        raise ValueError("facilities have no CRS")
    out = gpd.GeoDataFrame(
        {"name": facilities[name_column] if name_column else None},
        geometry=facilities.geometry,
        crs=facilities.crs,
    ).to_crs(4326)
    out.to_file(path, driver="GeoJSON")
    return Path(path)
