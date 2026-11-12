"""City configuration: one TOML file per city (``config/<city>.toml``), read with the standard-library ``tomllib``.

The configuration holds everything that is specific to a city and its data: file paths (relative to the project
root), CRS, the column layout of each source, reference years, building classes, OSM snapshot details, output
folders and city-specific validation expectations.

Two kinds of settings are kept apart:
- the method (in the code, not settings): the three objectives and their formulations, the nearest-open
  accessibility rule, the four-indicator equal-weight vulnerability index, census-grid allocation;
- model parameters (``[model_parameters]``, defaults = the settings used for Münster): facility budget P,
  walking speed, the vulnerability quantile that defines the least-advantaged group, solver time limits, MIP gaps
  and Stage 2 tolerances; plus ``[solver_resources]`` (threads, node files).

The active configuration is ``config/muenster.toml`` unless ``CHILDCARE_ACCESS_CONFIG`` names another file.

Checks, all reporting every problem at once through :class:`ConfigError`:
- structure and values (on load): required keys, types, value ranges, regular expressions, CRS;
- data (:func:`check_data`): input files exist (and match their checksums) and contain the configured columns.
"""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "muenster.toml"
CONFIG_ENV = "CHILDCARE_ACCESS_CONFIG"


class ConfigError(ValueError):
    """Invalid or incompatible city configuration (the message lists every problem found)."""


# Required keys and their types. A tuple of types means "any of these"; dict means a table of strings.
_SCHEMA: dict[str, dict[str, object]] = {
    "city": {"name": str, "scenario": str},
    "crs": {"projected": int},
    "sources.districts": {"file": str, "id_column": str, "name_column": str, "id_width": int},
    "sources.forecast": {
        "file": str,
        "year": int,
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
    },
    "sources.social": {
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
    },
    "sources.buildings": {
        "file": str,
        "sha256": str,
        "layer": str,
        "function_column": str,
        "object_type_column": str,
        "building_value": str,
        "part_value": str,
        "id_column": str,
        "mixed_use_code": str,
        "residential_codes": dict,
    },
    "sources.dormitories": {
        "exclude_min_share": (int, float),
        "report_min_share": (int, float),
        "student_pattern": str,
        "non_student_pattern": str,
        "decisions_file": str,
        "decisions_sha256": str,
    },
    "sources.census": {
        "file": str,
        "sha256": str,
        "member_suffix": str,
        "separator": str,
        "encoding": str,
        "crs": int,
        "cell_size_m": (int, float),
        "id_column": str,
        "x_column": str,
        "y_column": str,
        "count_columns": list,
    },
    "osm": {
        "cache_dir": str,
        "snapshot_timestamp": str,
        "snapshot_dir": str,
        "network_snapshot": str,
        "network_rule_set": str,
        "overpass_url": str,
        "user_agent": str,
        "kita_tags": dict,
    },
    "outputs": {"final_inputs": str, "final_run": str, "audit": str},
    "solver_resources": {
        "threads": int,
        "nodefile_start": (int, float),
        "nodefile_min_free_gb": (int, float),
        "nodefile_dir": str,
    },
}
# Optional tables: [model_parameters], [reproducibility], [validation] (city-specific expectations).

# Model parameters and their defaults (the settings used for the Münster case study).
PARAMETER_DEFAULTS: dict[str, object] = {
    "facility_budget": 3,
    "walking_speed_m_per_min": 83.33,
    "least_advantaged_quantile": 0.75,
    "time_limit_seconds": 900,
    "utilitarian_mip_gap": 0.01,
    "rawls_stage1_mip_gap": 0.001,
    "rawls_stage2_mip_gap": 0.01,
    "rawls_stage2_tolerances": [1e-6, 1e-4, 1e-3, 1e-2],
    "district_time_limit_seconds": 3600,
    "district_mip_gap": 1e-4,
}
# Keys allowed in a descriptive companion file <config>_metadata.toml (labels and provenance only).
METADATA_KEYS: dict[str, dict[str, type]] = {"demand": {"census_year": int, "census_source": str}}

# Input files checked by check_data(): config key -> (sha256 key or None).
_INPUT_FILES = {
    "sources.districts.file": None,
    "sources.forecast.file": None,
    "sources.social.file": None,
    "sources.buildings.file": "sources.buildings.sha256",
    "sources.dormitories.decisions_file": "sources.dormitories.decisions_sha256",
    "sources.census.file": "sources.census.sha256",
    "osm.network_snapshot": None,
}


@dataclass(frozen=True)
class CityConfig:
    """A loaded, structurally validated city configuration."""

    source: Path
    root: Path
    data: dict = field(repr=False)

    def get(self, dotted: str, default: Any = ...) -> Any:
        """Value at a dotted key such as ``"sources.forecast.year"``."""
        node = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise ConfigError(f"{self.source}: missing key [{dotted}]")
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Configured path, relative to the project root (as used by the scripts, run from the root)."""
        return Path(self.get(dotted))

    def resolve(self, dotted: str) -> Path:
        """Configured path made absolute against the project root (for existence checks)."""
        value = self.path(dotted)
        return value if value.is_absolute() else self.root / value

    @property
    def version(self) -> int:
        """Schema version of the file (1 when it states none)."""
        return self.data.get("schema", {}).get("version", 1)

    @property
    def parameters(self) -> dict[str, object]:
        """Model parameters: the ``[model_parameters]`` values over the defaults.

        In version 2 the least-advantaged quantile is set with the vulnerability method (``[vulnerability.<mode>]
        quantile``).
        """
        merged = {**PARAMETER_DEFAULTS, **self.data.get("model_parameters", {})}
        if self.version == 2:
            mode = self.data["vulnerability"]["mode"]
            merged["least_advantaged_quantile"] = (
                self.data["vulnerability"]
                .get(mode, {})
                .get("quantile", PARAMETER_DEFAULTS["least_advantaged_quantile"])
            )
        return merged

    @property
    def metadata(self) -> dict:
        """Descriptive metadata from the companion file ``<config stem>_metadata.toml`` (e.g.

        ``config/muenster_metadata.toml``), if present. Used only for labels and provenance, never for data loading,
        calculations, candidates or solver inputs.
        """
        path = self.source.with_name(f"{self.source.stem}_metadata.toml")
        if not path.is_file():
            return {}
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        unknown = [
            f"[{t}] {k}"
            for t, keys in data.items()
            for k in (keys if isinstance(keys, dict) else [None])
            if k not in METADATA_KEYS.get(t, {})
        ]
        wrong = [
            f"[{t}] {k}"
            for t, keys in data.items()
            if isinstance(keys, dict)
            for k, v in keys.items()
            if k in METADATA_KEYS.get(t, {}) and not isinstance(v, METADATA_KEYS[t][k])
        ]
        if unknown or wrong:
            raise ConfigError(
                f"{path}: only descriptive keys {METADATA_KEYS} are allowed; unknown {unknown}, wrong type {wrong}"
            )
        return data

    @property
    def methods(self) -> dict:
        """The selected methods and their metadata.

        Version 1 (the older layout) means: census-grid demand from the projected forecast, the four-indicator index,
        residential-building candidates. A version-1 file records no census year; it comes from the descriptive
        companion file, if any (``metadata``), else None.
        """
        d = self.data
        if self.version == 1:
            return {
                "demand": {
                    "mode": "census_hybrid",
                    "count_type": "projected",
                    "reference_year": d["sources"]["forecast"]["year"],
                    "census_year": self.metadata.get("demand", {}).get("census_year"),
                },
                "vulnerability": {
                    "mode": "four_indicator",
                    "quantile": self.parameters["least_advantaged_quantile"],
                },
                "candidates": {"source": "residential_buildings"},
                "facilities": {"source": "file" if d.get("osm", {}).get("kita_file") else "osm_cache"},
            }
        mode = d["demand"]["mode"]
        counts = d["demand"]["direct"] if mode == "direct" else d["population"]
        demand = {
            "mode": mode,
            "count_type": counts["count_type"],
            "reference_year": counts["reference_year"],
            "source": counts["source"],
            "census_year": d["demand"]["census"]["reference_year"] if mode == "census_hybrid" else None,
        }
        if mode == "direct":
            demand.update({k: counts.get(k) for k in ("geometry", "polygon_allocation", "boundary_rule")})
        vulnerability = {
            "mode": d["vulnerability"]["mode"],
            "quantile": self.parameters["least_advantaged_quantile"]
            if d["vulnerability"]["mode"] != "priority_districts"
            else None,
        }
        return {
            "demand": demand,
            "vulnerability": vulnerability,
            "candidates": {"source": d["candidates"]["source"]},
            "facilities": {"source": d["facilities"]["source"]},
        }


def _lookup(data: dict, dotted: str) -> Any:
    """Value at a dotted key of nested tables, or ``...`` if a part is missing."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return ...
        node = node[part]
    return node


def _check_structure(data: dict) -> list[str]:
    """Problems with the tables and keys of a configuration (missing or unknown ones)."""
    problems = []
    for section, keys in _SCHEMA.items():
        table = _lookup(data, section)
        if table is ... or not isinstance(table, dict):
            problems.append(f"missing table [{section}]")
            continue
        for key, kind in keys.items():
            if key not in table:
                problems.append(f"missing key [{section}] {key}")
            elif kind is dict:
                if not isinstance(table[key], dict) or not table[key]:
                    problems.append(f"[{section}] {key} must be a non-empty table")
            elif not isinstance(table[key], kind) or isinstance(table[key], bool):
                expected = " or ".join(k.__name__ for k in (kind if isinstance(kind, tuple) else (kind,)))
                problems.append(f"[{section}] {key} must be {expected}, got {type(table[key]).__name__}")
    return problems


def _check_values(data: dict) -> list[str]:
    """Problems with the values of a configuration (types, patterns, checksums, ranges)."""
    problems = []

    def get(dotted: str) -> Any:
        return _lookup(data, dotted)

    for key in ["sources.forecast.district_code_pattern", "sources.social.district_code_pattern"]:
        try:
            if re.compile(get(key)).groups < 1:
                problems.append(f"[{key}] needs one capture group for the district ID")
        except re.error as error:
            problems.append(f"[{key}] is not a valid regular expression: {error}")
    for key in ["sources.dormitories.student_pattern", "sources.dormitories.non_student_pattern"]:
        try:
            re.compile(get(key))
        except re.error as error:
            problems.append(f"[{key}] is not a valid regular expression: {error}")

    lower, upper = (
        get("sources.dormitories.report_min_share"),
        get("sources.dormitories.exclude_min_share"),
    )
    if not 0 <= lower <= upper <= 1:
        problems.append(
            f"dormitory shares must satisfy 0 <= report_min_share ({lower}) <= exclude_min_share ({upper}) <= 1"
        )
    if get("sources.districts.id_width") < 0:
        problems.append("[sources.districts] id_width must be >= 0")
    if get("sources.census.cell_size_m") <= 0:
        problems.append("[sources.census] cell_size_m must be > 0")
    if not get("sources.census.count_columns") or not all(
        isinstance(c, str) for c in get("sources.census.count_columns")
    ):
        problems.append("[sources.census] count_columns must be a non-empty list of column names")
    definitions = _lookup(data, "sources.social.indicator_definitions")
    if definitions is not ... and set(definitions) != set(get("sources.social.indicators")):
        problems.append(
            "[sources.social.indicator_definitions] must document exactly the indicators of [sources.social.indicators]"
        )
    if get("sources.buildings.mixed_use_code") not in get("sources.buildings.residential_codes"):
        problems.append("[sources.buildings] mixed_use_code must be one of residential_codes")
    for key in [
        "sources.forecast.age_groups",
        "sources.social.indicators",
        "osm.kita_tags",
        "sources.buildings.residential_codes",
    ]:
        if not all(isinstance(v, str) for v in get(key).values()):
            problems.append(f"[{key}] values must be strings")
    for key in [
        "sources.buildings.sha256",
        "sources.census.sha256",
        "sources.dormitories.decisions_sha256",
    ]:
        if get(key) and not re.fullmatch(r"[0-9a-f]{64}", get(key)):
            problems.append(f"[{key}] must be a 64-character lowercase SHA-256 or empty")
    problems += _check_parameters(data)
    if get("solver_resources.threads") < 0 or get("solver_resources.nodefile_start") < 0:
        problems.append("[solver_resources] threads and nodefile_start must be >= 0 (0 = Gurobi default)")
    problems += _check_crs(get("crs.projected"), "crs.projected", projected=True)
    problems += _check_crs(get("sources.census.crs"), "sources.census.crs", projected=True)
    return problems


def _check_parameters(data: dict) -> list[str]:
    """Problems with the ``[model_parameters]`` table (only documented overrides are allowed)."""
    problems = []
    given = data.get("model_parameters", {})
    unknown = sorted(set(given) - set(PARAMETER_DEFAULTS))
    if unknown:
        problems.append(f"[model_parameters] unknown parameter(s) {unknown}; allowed: {sorted(PARAMETER_DEFAULTS)}")
    for key, value in given.items():
        default = PARAMETER_DEFAULTS.get(key)
        if default is None:
            continue
        if isinstance(default, list):
            if not (
                isinstance(value, list)
                and value
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in value)
            ):
                problems.append(f"[model_parameters] {key} must be a non-empty list of positive numbers")
        elif isinstance(default, int) and not isinstance(default, bool) and key == "facility_budget":
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                problems.append("[model_parameters] facility_budget must be an integer >= 1")
        elif not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            problems.append(f"[model_parameters] {key} must be a non-negative number")
    merged = {**PARAMETER_DEFAULTS, **given}
    if not problems:
        if not 0 < merged["least_advantaged_quantile"] < 1:
            problems.append("[model_parameters] least_advantaged_quantile must lie strictly between 0 and 1")
        if (
            merged["walking_speed_m_per_min"] <= 0
            or merged["time_limit_seconds"] <= 0
            or merged["district_time_limit_seconds"] <= 0
        ):
            problems.append("[model_parameters] walking speed and time limits must be > 0")
    return problems


def _check_v2(data: dict) -> list[str]:
    """Version 2: the tables of the selected methods, the model parameters and the CRS (see :mod:`requirements`)."""
    from .requirements import structural_problems

    problems = structural_problems(data)
    if problems:
        return problems
    if "least_advantaged_quantile" in data.get("model_parameters", {}):
        problems.append(
            "[model_parameters] least_advantaged_quantile is set with the vulnerability method in version 2 "
            "([vulnerability.<mode>] quantile)"
        )
    problems += _check_parameters(data)
    if data["solver_resources"]["threads"] < 0 or data["solver_resources"]["nodefile_start"] < 0:
        problems.append("[solver_resources] threads and nodefile_start must be >= 0 (0 = Gurobi default)")
    problems += _check_crs(data["crs"]["projected"], "crs.projected", projected=True)
    if data["demand"]["mode"] == "census_hybrid":
        problems += _check_crs(data["demand"]["census"]["crs"], "demand.census.crs", projected=True)
    return problems


def _check_crs(code: object, key: str, projected: bool) -> list[str]:
    """Problems with one CRS key (unknown code, or geographic where metres are needed)."""
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        crs = CRS.from_user_input(code)
    except CRSError as error:
        return [f"[{key}] {code!r} is not a valid CRS: {error}"]
    if projected and not crs.is_projected:
        return [f"[{key}] EPSG:{code} is not a projected CRS (distances in metres are needed)"]
    units = {axis.unit_name for axis in crs.axis_info}
    if projected and not units <= {"metre"}:
        return [f"[{key}] EPSG:{code} does not use metres (units: {sorted(units)})"]
    return []


def load_city_config(path: str | Path | None = None) -> CityConfig:
    """Read and structurally validate a city configuration. Raises :class:`ConfigError` listing every problem."""
    source = Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG)
    if not source.is_absolute():
        source = (Path.cwd() / source) if (Path.cwd() / source).exists() else PROJECT_ROOT / source
    if not source.is_file():
        raise ConfigError(f"city configuration not found: {source} (set {CONFIG_ENV} or create config/<city>.toml)")
    try:
        data = tomllib.loads(source.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{source}: invalid TOML: {error}") from error
    version = data.get("schema", {}).get("version", 1)
    if version == 1:
        problems = _check_structure(data)
        if not problems:
            problems = _check_values(data)
    elif version == 2:
        problems = _check_v2(data)
    else:
        problems = [f"[schema] version must be 1 or 2, got {version!r}"]
    if problems:
        raise ConfigError(f"{source}: {len(problems)} problem(s):\n  - " + "\n  - ".join(problems))
    return CityConfig(source=source, root=PROJECT_ROOT, data=data)


@lru_cache(maxsize=1)
def active_config() -> CityConfig:
    """The configuration selected by ``CHILDCARE_ACCESS_CONFIG`` (default: Münster), loaded once per process."""
    return load_city_config()


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_data(config: CityConfig, checksums: bool = True, columns: bool = True) -> dict[str, str]:
    """Check that every input file exists, matches its recorded checksum and has the configured columns.

    Returns {config key: "ok"}; raises :class:`ConfigError` listing every missing file, checksum mismatch and
    missing column. Reads only file headers (and hashes the checksummed files when ``checksums``).
    """
    problems, report = [], {}
    for key, sha_key in _INPUT_FILES.items():
        path = config.resolve(key)
        if not path.is_file():
            problems.append(f"missing dataset [{key}]: {config.path(key)}")
            continue
        expected = config.get(sha_key) if sha_key else ""
        if checksums and expected and sha256_of(path) != expected:
            problems.append(f"checksum mismatch [{key}]: {config.path(key)} differs from the recorded SHA-256")
            continue
        report[key] = "ok"
    if columns:
        problems += _check_columns(config, report)
    if problems:
        raise ConfigError(f"{config.source}: {len(problems)} data problem(s):\n  - " + "\n  - ".join(problems))
    return report


def _check_columns(config: CityConfig, present: dict[str, str]) -> list[str]:
    """Problems with the columns of the input files that the selected methods read."""
    import pandas as pd
    import pyogrio

    problems = []

    def need(key: str, found: Iterable[str], wanted: list[str]) -> None:
        """Record the wanted columns missing from one file."""
        missing = [c for c in wanted if c not in set(found)]
        if missing:
            problems.append(f"[{key}] {config.path(key)} lacks configured column(s) {missing}")

    if "sources.districts.file" in present:
        info = pyogrio.read_info(config.resolve("sources.districts.file"))
        need(
            "sources.districts.file",
            info["fields"],
            [
                config.get("sources.districts.id_column"),
                config.get("sources.districts.name_column"),
            ],
        )
        if info["crs"] is None:
            problems.append(f"[sources.districts.file] {config.path('sources.districts.file')} has no CRS")
    for section, columns in [
        ("forecast", ["area_column", "year_column", "sex_column", "age_column", "value_column"]),
        ("social", ["area_column", "time_column", "indicator_column", "value_column"]),
    ]:
        key = f"sources.{section}.file"
        if key in present:
            header = pd.read_csv(
                config.resolve(key),
                sep=config.get(f"sources.{section}.separator"),
                encoding=config.get(f"sources.{section}.encoding"),
                nrows=0,
            ).columns
            need(key, header, [config.get(f"sources.{section}.{c}") for c in columns])
    if "sources.buildings.file" in present:
        path = config.resolve("sources.buildings.file")
        layers = [name for name, _ in pyogrio.list_layers(path)]
        layer = config.get("sources.buildings.layer")
        if layer not in layers:
            problems.append(
                f"[sources.buildings.layer] layer {layer!r} not in {config.path('sources.buildings.file')} (layers: "
                f"{layers})"
            )
        else:
            need(
                "sources.buildings.file",
                pyogrio.read_info(path, layer=layer)["fields"],
                [config.get(f"sources.buildings.{c}") for c in ["function_column", "object_type_column", "id_column"]],
            )
    if "sources.census.file" in present:
        import io
        import zipfile

        with zipfile.ZipFile(config.resolve("sources.census.file")) as archive:
            members = [n for n in archive.namelist() if n.endswith(config.get("sources.census.member_suffix"))]
            if not members:
                problems.append(
                    "[sources.census.member_suffix] no file ending in "
                    f"{config.get('sources.census.member_suffix')!r} in the zip"
                )
            else:
                with archive.open(members[0]) as handle:
                    header = pd.read_csv(
                        io.TextIOWrapper(handle, encoding=config.get("sources.census.encoding")),
                        sep=config.get("sources.census.separator"),
                        nrows=0,
                    ).columns
                need(
                    "sources.census.file",
                    header,
                    [
                        config.get("sources.census.id_column"),
                        config.get("sources.census.x_column"),
                        config.get("sources.census.y_column"),
                        *config.get("sources.census.count_columns"),
                    ],
                )
    return problems


def default_nodefile_dir(config: CityConfig) -> Path:
    """Gurobi node-file folder: the configured one, else a per-user default."""
    configured = config.get("solver_resources.nodefile_dir")
    if configured:
        return Path(configured)
    base = (
        Path.home() / "AppData/Local"
        if os.name == "nt"
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    )
    return base / "gurobi_nodefiles" / config.get("city.scenario")


def check_city(path: str | Path) -> tuple[bool, list[str]]:
    """Full check of a city configuration for ``childcare-access check``.

    Checks the structure and values, the data files and their columns, and leftover template placeholders. Never
    raises for a city problem.

    Returns:
        (ok, report lines).
    """
    lines, ok = [], True
    try:
        config = load_city_config(path)
    except ConfigError as error:
        return False, [f"FAIL configuration: {error}"]
    lines.append(f"PASS configuration: {config.source.name}")
    placeholders = [
        line.strip()
        for line in config.source.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#") and ("<-- REPLACE" in line or "YYYY" in line)
    ]
    if placeholders:
        ok = False
        lines.append(f"FAIL template placeholders left ({len(placeholders)}): " + "; ".join(placeholders[:5]))
    if config.version == 2:
        from .requirements import column_problems, file_problems, input_files

        methods = config.methods
        lines.append("methods: " + "; ".join(f"{k} = {v.get('mode', v.get('source'))}" for k, v in methods.items()))
        missing = file_problems(config.data, config.root) + column_problems(config.data, config.root)
        if missing:
            ok = False
            lines.append(f"FAIL data: {len(missing)} problem(s):\n  - " + "\n  - ".join(missing))
        else:
            lines.append(
                f"PASS data: the {len(input_files(config.data))} input files of the selected methods are present "
                "with their configured columns"
            )
        lines.append("model parameters: " + ", ".join(f"{k}={v}" for k, v in config.parameters.items()))
        return ok, lines
    try:
        report = check_data(config)
        lines.append(f"PASS data: {len(report)} input files present, checksums and configured columns as expected")
    except ConfigError as error:
        ok = False
        lines.append(f"FAIL data: {error}")
    lines.append("model parameters: " + ", ".join(f"{k}={v}" for k, v in config.parameters.items()))
    definitions = config.get("sources.social.indicator_definitions", None)
    lines.append(
        "indicator definitions: "
        + (
            "documented"
            if definitions
            else "NOT documented - add [sources.social.indicator_definitions] (source, definition, deviations)"
        )
    )
    return ok, lines
