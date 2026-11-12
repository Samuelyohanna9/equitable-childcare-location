"""Version-2 configurations of the synthetic city (test helper; not a test module).

``v2_config`` turns the committed version-1 example (config/synthetic_city.toml), pointed at freshly generated data,
into a version-2 file with the requested methods. The same raw data serve every method, so builds can be compared.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from test_workflow_synthetic import write_city


def to_toml(data: dict) -> str:
    """Minimal TOML writer for nested tables of strings, numbers, booleans and lists."""

    def value(v):
        return json.dumps(v) if not isinstance(v, bool) else ("true" if v else "false")

    lines, tables = [], []

    def walk(table: dict, prefix: str) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        if prefix:
            tables.append((prefix, scalars))
        else:
            lines.extend(f"{k} = {value(v)}" for k, v in scalars.items())
        for k, v in table.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}.{k}" if prefix else k)

    walk(data, "")
    for name, scalars in tables:
        lines.append(f"\n[{name}]")
        lines.extend(f"{k} = {value(v)}" for k, v in scalars.items())
    return "\n".join(lines) + "\n"


def base(root: Path) -> dict:
    """The synthetic city as a version-2 dict: census-grid demand, four-indicator index, residential candidates."""
    v1 = tomllib.loads(write_city(root).read_text(encoding="utf-8"))
    parameters = {k: v for k, v in v1["model_parameters"].items() if k != "least_advantaged_quantile"}
    social = v1["sources"]["social"]
    return {
        "schema": {"version": 2},
        "city": v1["city"],
        "crs": v1["crs"],
        "workspace": v1["workspace"],
        "sources": {"districts": v1["sources"]["districts"]},
        "osm": {k: v1["osm"][k] for k in ("network_snapshot", "snapshot_timestamp", "network_rule_set", "cache_dir")},
        "facilities": {
            "source": "file",
            "file": v1["osm"]["kita_file"],
            "id_column": "facility_id",
        },
        "demand": {
            "mode": "census_hybrid",
            "census": {**v1["sources"]["census"], "reference_year": 2022},
        },
        "population": {
            **v1["sources"]["forecast"],
            "layout": "long",
            "district_id_column": "district_id",
            "count_type": "projected",
            "reference_year": v1["sources"]["forecast"]["year"],
            "source": "synthetic forecast (make_city.py)",
            "age_0_2": "0-2",
            "age_3_5": "3-5",
        },
        "buildings": {
            **v1["sources"]["buildings"],
            "dormitories": {**v1["sources"]["dormitories"], "building_tags_source": "none"},
        },
        "vulnerability": {
            "mode": "four_indicator",
            "four_indicator": {**social, "quantile": v1["model_parameters"]["least_advantaged_quantile"]},
        },
        "candidates": {"source": "residential_buildings"},
        "model_parameters": parameters,
        "solver_resources": v1["solver_resources"],
    }


def write(data: dict, path: Path) -> Path:
    path.write_text(to_toml(data), encoding="utf-8")
    return path
