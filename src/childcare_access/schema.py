"""Standard tables between the city data and the model-input assembly.

Every demand, vulnerability, facility and candidate source produces these tables; the assembly (snapping, demand
locations, access, relations, screen) reads only them. ``check`` reports every problem of a table at once.

| Table | Required columns | Optional columns |
|---|---|---|
| districts | district_id, district_name, geometry | |
| child_totals | district_id, children_0_5 | age_0_2, age_3_5 (both or neither; 0-5 = their sum) |
| priority_groups | district_id, district_name, priority, group | index_value, rank |
| demand_origins | origin_id, district_id, x, y, children_0_5, origin_type, source_id | age_0_2, age_3_5 |
| facilities | facility_id, x, y, source | |
| candidates | site_id, x, y, source, eligible, experimental | |

Coordinates are in the projected CRS of the configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOLERANCE = 1e-9  # absolute tolerance (children) for age sums and conservation
ORIGIN_TYPES = (
    "building",
    "census_cell_centre",
    "supplied_point",
    "polygon_building",
    "polygon_node",
)
COLUMNS = {
    "districts": {"district_id": "id", "district_name": "text", "geometry": "geometry"},
    "child_totals": {"district_id": "id", "children_0_5": "count"},
    "priority_groups": {
        "district_id": "id",
        "district_name": "text",
        "priority": "bool",
        "group": "text",
    },
    "demand_origins": {
        "origin_id": "id",
        "district_id": "id",
        "x": "number",
        "y": "number",
        "children_0_5": "count",
        "origin_type": "text",
        "source_id": "text",
    },
    "facilities": {"facility_id": "id", "x": "number", "y": "number", "source": "text"},
    "candidates": {
        "site_id": "id",
        "x": "number",
        "y": "number",
        "source": "text",
        "eligible": "bool",
        "experimental": "bool",
    },
}
KEYS = {
    "districts": "district_id",
    "child_totals": "district_id",
    "priority_groups": "district_id",
    "demand_origins": "origin_id",
    "facilities": "facility_id",
    "candidates": "site_id",
}


class SchemaError(ValueError):
    """A standard table violates its schema (the message lists every problem)."""


def _column(frame: pd.DataFrame, column: str, kind: str) -> list[str]:
    """Problems with one column of a schema table (missing values, wrong type)."""
    values = frame[column]
    if kind == "geometry":
        return [f"{column}: {int(values.isna().sum())} missing geometries"] if values.isna().any() else []
    if values.isna().any():
        return [f"{column}: {int(values.isna().sum())} missing values"]
    if kind == "bool":
        return [] if pd.api.types.is_bool_dtype(values) else [f"{column}: must be true/false, got {values.dtype}"]
    if kind in ("number", "count"):
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            return [f"{column}: must be numeric, got {values.dtype}"]
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            return [f"{column}: non-finite values"]
        if kind == "count" and (values < 0).any():
            return [f"{column}: {int((values < 0).sum())} negative counts"]
    return []


def check(name: str, frame: pd.DataFrame, districts: pd.DataFrame | None = None) -> None:
    """Raise :class:`SchemaError` if ``frame`` is not a valid ``name`` table.

    ``districts`` is the districts table, for the district-coverage checks.
    """
    problems = [f"missing column {c}" for c in COLUMNS[name] if c not in frame.columns]
    if not problems:
        for column, kind in COLUMNS[name].items():
            problems += _column(frame, column, kind)
        key = KEYS[name]
        if frame[key].duplicated().any():
            problems.append(
                f"{key}: duplicates {frame.loc[frame[key].duplicated(), key].astype(str).unique()[:5].tolist()}"
            )
    ages = [c for c in ("age_0_2", "age_3_5") if c in frame.columns]
    if name in ("child_totals", "demand_origins") and ages and not problems:
        if len(ages) == 1:
            problems.append(f"only {ages[0]} given: give both age groups or neither")
        else:
            for column in ages:
                problems += _column(frame, column, "count")
            if not problems:
                gap = (frame["age_0_2"] + frame["age_3_5"] - frame["children_0_5"]).abs()
                if (gap > TOLERANCE).any():
                    problems.append(
                        f"age_0_2 + age_3_5 differs from children_0_5 in {int((gap > TOLERANCE).sum())} rows "
                        f"(max {gap.max():.3g})"
                    )
    if name == "demand_origins" and not problems:
        unknown = sorted(set(frame["origin_type"]) - set(ORIGIN_TYPES))
        if unknown:
            problems.append(f"origin_type: unknown {unknown}; allowed {list(ORIGIN_TYPES)}")
    if name == "districts" and not problems and getattr(frame, "crs", None) is not None and not frame.crs.is_projected:
        problems.append("geometry: not in a projected CRS")
    if districts is not None and name != "districts" and "district_id" in frame.columns:
        known = set(districts["district_id"])
        unknown = sorted(set(frame["district_id"]) - known)
        if unknown:
            problems.append(f"district_id not in the districts: {unknown[:5]}")
        if name in ("child_totals", "priority_groups"):
            absent = sorted(known - set(frame["district_id"]))
            if absent:
                problems.append(f"districts without a row: {absent[:5]}")
    if problems:
        raise SchemaError(f"{name}: {len(problems)} problem(s):\n  - " + "\n  - ".join(problems))


def conservation(origins: pd.DataFrame, totals: pd.DataFrame, column: str = "children_0_5") -> pd.Series:
    """Per district: allocated origins minus the district total (zero within ``TOLERANCE`` when conserved)."""
    allocated = origins.groupby("district_id")[column].sum()
    target = totals.set_index("district_id")[column]
    return allocated.reindex(target.index, fill_value=0.0) - target
