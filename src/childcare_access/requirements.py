"""Inputs required by the selected methods of a version-2 city configuration.

A v2 configuration names its methods explicitly (``[demand] mode``, ``[vulnerability] mode``, ``[candidates] source``,
``[facilities] source``). Only the tables those methods use are required. Every problem is reported at once, with
the explicit alternative where one exists; a method is never switched because its data are missing.

The network keys stay in [osm] under their version-1 names (pedestrian_network reads them on import).
Method tables keep the column layout of the version-1 tables they correspond to (``[population]`` with
``layout = "long"`` as ``[sources.forecast]``, ``[demand.census]`` as ``[sources.census]``, ``[buildings]`` as
``[sources.buildings]``, ``[vulnerability.four_indicator]`` as ``[sources.social]``), so the same loaders read
both layouts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEMAND_MODES = ("census_hybrid", "building_area", "direct")
VULNERABILITY_MODES = ("four_indicator", "external_index", "priority_districts")
CANDIDATE_SOURCES = ("residential_buildings", "file", "demand_nodes")
FACILITY_SOURCES = ("file", "osm_cache")
COUNT_TYPES = ("observed", "projected")

# Tables every v2 city needs: table -> {key: type}.
ESSENTIAL = {
    "city": {"name": str, "scenario": str},
    "crs": {"projected": int},
    "workspace": {"root": str},
    "sources.districts": {"file": str, "id_column": str, "name_column": str, "id_width": int},
    "osm": {"network_snapshot": str, "snapshot_timestamp": str, "network_rule_set": str},
    "facilities": {"source": str},
    "demand": {"mode": str},
    "vulnerability": {"mode": str},
    "candidates": {"source": str},
    "solver_resources": {
        "threads": int,
        "nodefile_start": (int, float),
        "nodefile_min_free_gb": (int, float),
        "nodefile_dir": str,
    },
}
POPULATION = {
    "file": str,
    "layout": str,
    "district_id_column": str,
    "count_type": str,
    "reference_year": int,
    "source": str,
}
POPULATION_LONG = {
    "separator": str,
    "encoding": str,
    "area_column": str,
    "year_column": str,
    "sex_column": str,
    "age_column": str,
    "value_column": str,
    "all_sexes_value": str,
    "city_total_marker": str,
    "district_code_pattern": str,
    "age_groups": dict,
}
CENSUS = {
    "file": str,
    "reference_year": int,
    "member_suffix": str,
    "separator": str,
    "encoding": str,
    "crs": int,
    "cell_size_m": (int, float),
    "id_column": str,
    "x_column": str,
    "y_column": str,
    "count_columns": list,
}
BUILDINGS = {
    "file": str,
    "layer": str,
    "id_column": str,
    "function_column": str,
    "residential_codes": dict,
    "mixed_use_code": str,
}
DIRECT = {
    "geometry": str,
    "file": str,
    "id_column": str,
    "count_type": str,
    "reference_year": int,
    "source": str,
}
FOUR_INDICATOR = {
    "file": str,
    "year": int,
    "separator": str,
    "encoding": str,
    "decimal": str,
    "area_column": str,
    "time_column": str,
    "indicator_column": str,
    "value_column": str,
    "district_code_pattern": str,
    "indicators": dict,
    "indicator_definitions": dict,
}
EXTERNAL_INDEX = {
    "file": str,
    "district_id_column": str,
    "value_column": str,
    "name": str,
    "definition": str,
    "higher_is_more_vulnerable": bool,
    "quantile": float,
    "ties": str,
    "missing": str,
}
PRIORITY = {"file": str, "district_id_column": str, "label": str, "definition": str}
CANDIDATE_FILE = {"file": str, "id_column": str}
DOCUMENTING = ("source", "name", "definition", "label")  # keys that may not be left empty

# Said when a method's table is missing: the explicit alternatives (never applied automatically).
ALTERNATIVES = {
    "demand.census": "no census grid? select demand.mode = 'building_area' or 'direct' explicitly; "
    "this changes the demand definition",
    "population": "no district child totals? select demand.mode = 'direct' with supplied child counts",
    "vulnerability.four_indicator": "no four indicators? select vulnerability.mode = 'external_index' or "
    "'priority_districts' explicitly; this changes the priority-group definition",
}


def _get(data: dict, dotted: str) -> Any:
    """Value at a dotted key of nested tables, or ``...`` if a part is missing."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return ...
        node = node[part]
    return node


def _typed(data: dict, table: str, keys: dict, why: str) -> list[str]:
    """Missing or mistyped keys of ``table``; ``why`` names the method that needs it."""
    found = _get(data, table)
    if not isinstance(found, dict):
        hint = f"; {ALTERNATIVES[table]}" if table in ALTERNATIVES else ""
        return [f"missing table [{table}] (needed by {why}){hint}"]
    problems = []
    for key, kind in keys.items():
        value = found.get(key, ...)
        if value is ...:
            problems.append(f"missing key [{table}] {key} (needed by {why})")
        elif kind is dict:
            if not isinstance(value, dict) or not value:
                problems.append(f"[{table}] {key} must be a non-empty table")
        elif kind is bool:
            if not isinstance(value, bool):
                problems.append(f"[{table}] {key} must be true or false")
        elif kind is float:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"[{table}] {key} must be a number")
        elif not isinstance(value, kind) or isinstance(value, bool):
            expected = " or ".join(k.__name__ for k in (kind if isinstance(kind, tuple) else (kind,)))
            problems.append(f"[{table}] {key} must be {expected}, got {type(value).__name__}")
        elif key in DOCUMENTING and not value.strip():
            problems.append(f"[{table}] {key} must not be empty (it documents the data)")
    return problems


def _choice(data: dict, dotted: str, allowed: tuple) -> list[str]:
    """Problem if the value is not one of the allowed ones."""
    value = _get(data, dotted)
    return (
        []
        if value in allowed
        else [f"[{dotted}] must be one of {list(allowed)}, got {None if value is ... else value!r}"]
    )


def _counts(data: dict, table: str) -> list[str]:
    """Child counts: ``children_0_5``, or both ``age_0_2`` and ``age_3_5`` (0-5 = their sum), never one age group."""
    found = _get(data, table)
    given = {k: bool(found.get(k)) for k in ("children_0_5", "age_0_2", "age_3_5")}
    problems = _choice(data, f"{table}.count_type", COUNT_TYPES)
    if given["age_0_2"] != given["age_3_5"]:
        problems.append(f"[{table}] give both age_0_2 and age_3_5 or neither (one age group alone cannot be checked)")
    elif not given["children_0_5"] and not given["age_0_2"]:
        problems.append(f"[{table}] give children_0_5, or age_0_2 and age_3_5")
    return problems


def _quantile(data: dict, dotted: str) -> list[str]:
    """Problem if the value is not strictly between 0 and 1."""
    value = _get(data, dotted)
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < 1
    return [] if ok else [f"[{dotted}] must lie strictly between 0 and 1"]


def _pattern(data: dict, dotted: str) -> list[str]:
    """Problem if the value is not a regular expression with at least one group."""
    try:
        return (
            []
            if re.compile(_get(data, dotted)).groups >= 1
            else [f"[{dotted}] needs one capture group for the district ID"]
        )
    except re.error as error:
        return [f"[{dotted}] is not a valid regular expression: {error}"]


def _population(data: dict, why: str) -> list[str]:
    """Problems with the ``[population]`` table."""
    problems = _typed(data, "population", POPULATION, why)
    if problems:
        return problems
    problems = _counts(data, "population") + _choice(data, "population.layout", ("wide", "long"))
    if data["population"]["layout"] == "long":
        problems += _typed(data, "population", POPULATION_LONG, "population.layout = 'long'")
        if not problems:
            problems += _pattern(data, "population.district_code_pattern")
    return problems


def structural_problems(data: dict) -> list[str]:
    """Every missing, mistyped or invalid key of a v2 configuration for its selected methods."""
    problems = []
    for table, keys in ESSENTIAL.items():
        problems += _typed(data, table, keys, "every city")
    if problems:
        return problems
    problems += _choice(data, "demand.mode", DEMAND_MODES) + _choice(data, "vulnerability.mode", VULNERABILITY_MODES)
    problems += _choice(data, "candidates.source", CANDIDATE_SOURCES) + _choice(
        data, "facilities.source", FACILITY_SOURCES
    )
    if problems:
        return problems
    demand, vulnerability = data["demand"]["mode"], data["vulnerability"]["mode"]
    candidates, facilities = data["candidates"]["source"], data["facilities"]["source"]

    if facilities == "file":
        problems += _typed(data, "facilities", {"file": str, "id_column": str}, "facilities.source = 'file'")
    else:
        problems += _typed(data, "facilities", {"osm_tags": dict}, "facilities.source = 'osm_cache'")
        problems += _typed(data, "osm", {"cache_dir": str}, "facilities.source = 'osm_cache'")

    why = f"demand.mode = '{demand}'"
    if demand in ("census_hybrid", "building_area"):
        problems += _population(data, why) + _typed(data, "buildings", BUILDINGS, why)
    if demand == "census_hybrid":
        census = _typed(data, "demand.census", CENSUS, why)
        problems += census or (
            []
            if data["demand"]["census"]["count_columns"]
            else ["[demand.census] count_columns must name at least one column"]
        )
    if demand == "direct":
        direct = _typed(data, "demand.direct", DIRECT, why)
        problems += direct
        if not direct:
            problems += _counts(data, "demand.direct") + _choice(data, "demand.direct.geometry", ("points", "polygons"))
            if data["demand"]["direct"]["geometry"] == "polygons":
                problems += _choice(data, "demand.direct.polygon_allocation", ("building_area", "network_nodes"))
                problems += _choice(data, "demand.direct.boundary_rule", ("reject", "split"))
                if data["demand"]["direct"].get("polygon_allocation") == "building_area":
                    problems += _typed(data, "buildings", BUILDINGS, "polygon_allocation = 'building_area'")
        if isinstance(_get(data, "population"), dict):
            problems += _population(data, "the optional [population] consistency check")
        if _get(data, "demand.direct.consistency") is not ...:
            problems += _choice(data, "demand.direct.consistency", ("report", "error"))
        tolerance = _get(data, "demand.direct.consistency_tolerance")
        if tolerance is not ... and (
            not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0
        ):
            problems.append("[demand.direct] consistency_tolerance must be a number >= 0 (children per district)")
        if candidates == "residential_buildings":
            problems += _typed(data, "buildings", BUILDINGS, "candidates.source = 'residential_buildings'")

    if candidates == "file":
        problems += _typed(data, "candidates", CANDIDATE_FILE, "candidates.source = 'file'")
    if _get(data, "candidates.polygon_point") not in (..., ""):
        problems += _choice(data, "candidates.polygon_point", ("representative_point", "centroid"))
    values = _get(data, "candidates.eligible_values")
    if values is not ... and not isinstance(values, list):
        problems.append("[candidates] eligible_values must be a list")
    if _get(data, "candidates.eligible_column") not in (..., "") and not values:
        problems.append(
            "[candidates] eligible_column is set: eligible_values must list the values that count as eligible"
        )
    if _get(data, "candidates.boundary_site") is not ...:
        problems += _choice(data, "candidates.boundary_site", ("error", "lowest_id"))
    if _get(data, "candidates.shared_node") is not ...:
        problems += _choice(data, "candidates.shared_node", ("error", "keep_first"))
    snap = _get(data, "candidates.max_snap_m")
    if snap is not ... and (not isinstance(snap, (int, float)) or isinstance(snap, bool) or snap <= 0):
        problems.append("[candidates] max_snap_m must be > 0")

    why = f"vulnerability.mode = '{vulnerability}'"
    if vulnerability == "four_indicator":
        four = _typed(data, "vulnerability.four_indicator", FOUR_INDICATOR, why)
        problems += four
        if not four:
            table = data["vulnerability"]["four_indicator"]
            problems += _pattern(data, "vulnerability.four_indicator.district_code_pattern")
            if set(table["indicator_definitions"]) != set(table["indicators"]):
                problems.append(
                    "[vulnerability.four_indicator] indicator_definitions must document exactly the indicators"
                )
            if "quantile" in table:
                problems += _quantile(data, "vulnerability.four_indicator.quantile")
    if vulnerability == "external_index":
        external = _typed(data, "vulnerability.external_index", EXTERNAL_INDEX, why)
        problems += external
        if not external:
            problems += _quantile(data, "vulnerability.external_index.quantile")
            problems += _choice(data, "vulnerability.external_index.ties", ("include",))
            problems += _choice(data, "vulnerability.external_index.missing", ("error", "not_priority"))
    if vulnerability == "priority_districts":
        problems += _typed(data, "vulnerability.priority_districts", PRIORITY, why)
    for table in ("vulnerability.external_index", "vulnerability.priority_districts"):
        count = _get(data, f"{table}.expected_count")
        if count is not ... and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
            problems.append(f"[{table}] expected_count must be an integer >= 0 (0 = not checked)")
    return problems


def input_files(data: dict) -> list[tuple[str, str]]:
    """(table, key) of every input file the selected methods read."""
    demand, direct = data["demand"]["mode"], data["demand"].get("direct", {})
    tables = ["sources.districts", "osm"]
    if demand in ("census_hybrid", "building_area") or (demand == "direct" and "population" in data):
        tables.append("population")
    if demand == "census_hybrid":
        tables.append("demand.census")
    if demand == "direct":
        tables.append("demand.direct")
    polygons_by_area = direct.get("geometry") == "polygons" and direct.get("polygon_allocation") == "building_area"
    if demand != "direct" or polygons_by_area or data["candidates"]["source"] == "residential_buildings":
        tables.append("buildings")
    tables.append(f"vulnerability.{data['vulnerability']['mode']}")
    if data["facilities"]["source"] == "file":
        tables.append("facilities")
    if data["candidates"]["source"] == "file":
        tables.append("candidates")
    return [(t, "network_snapshot" if t == "osm" else "file") for t in tables]


def column_problems(data: dict, root: Path) -> list[str]:
    """Configured columns missing from the (present) input files of the selected methods; reads headers only."""
    import pandas as pd
    import pyogrio

    def resolve(value: str) -> Path:
        """A configured path, relative to the project root unless it is absolute."""
        return Path(value) if Path(value).is_absolute() else root / value

    problems = []

    def need(table: str, found: Iterable[str], wanted: Iterable[str]) -> None:
        """Record the configured columns missing from one table."""
        missing = [c for c in wanted if c and c not in set(found)]
        if missing:
            problems.append(f"[{table}] {_get(data, table + '.file')} lacks configured column(s) {missing}")

    counts = ("children_0_5", "age_0_2", "age_3_5")
    population = _get(data, "population")
    if isinstance(population, dict) and population.get("layout") == "wide" and resolve(population["file"]).is_file():
        header = pd.read_csv(
            resolve(population["file"]),
            sep=population.get("separator", ","),
            encoding=population.get("encoding", "utf-8"),
            nrows=0,
        ).columns
        need(
            "population",
            header,
            [population["district_id_column"], *(population.get(c, "") for c in counts)],
        )
    direct = _get(data, "demand.direct")
    if data["demand"]["mode"] == "direct" and resolve(direct["file"]).is_file():
        info = pyogrio.read_info(resolve(direct["file"]))
        need(
            "demand.direct",
            info["fields"],
            [
                direct["id_column"],
                direct.get("district_column", ""),
                *(direct.get(c, "") for c in counts),
            ],
        )
        if info["crs"] is None:
            problems.append(f"[demand.direct] {direct['file']} has no CRS")
    buildings = _get(data, "buildings")
    if (
        isinstance(buildings, dict)
        and ("buildings", "file") in input_files(data)
        and resolve(buildings["file"]).is_file()
    ):
        layers = [name for name, _ in pyogrio.list_layers(resolve(buildings["file"]))]
        if buildings["layer"] not in layers:
            problems.append(f"[buildings] layer {buildings['layer']!r} not in {buildings['file']} (layers: {layers})")
        else:
            need(
                "buildings",
                pyogrio.read_info(resolve(buildings["file"]), layer=buildings["layer"])["fields"],
                [
                    buildings["id_column"],
                    buildings["function_column"],
                    buildings.get("object_type_column", ""),
                ],
            )
    return problems


def file_problems(data: dict, root: Path) -> list[str]:
    """Missing input files of the selected methods (paths relative to ``root``)."""
    problems = []
    for table, key in input_files(data):
        value = _get(data, f"{table}.{key}")
        path = Path(value)
        if not (path if path.is_absolute() else root / path).is_file():
            problems.append(f"missing dataset [{table}] {key} = {value!r}")
    return problems
