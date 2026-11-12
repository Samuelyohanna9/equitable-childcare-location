"""City settings of the active configuration (``config/<city>.toml``; see :mod:`childcare_access.config`).

Paths, source layouts, years, building codes, dormitory rules, census grid, OSM tags, solver resources and the
district model's parameters, read once from the configuration.

A version-2 configuration keeps each method table in its own place ([population], [demand.census], [buildings],
[vulnerability.four_indicator], [facilities]); a table its selected methods do not use may be absent, and the
names read from it are then None (or empty). Version-1 configurations (Münster, the demo city) keep the older layout.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import ConfigError, active_config, default_nodefile_dir
from .core import PARAMETERS, PROJECTED_CRS, VULNERABILITY_COLUMNS  # noqa: F401  (PROJECTED_CRS re-exported)

CONFIG = active_config()
V2 = CONFIG.version == 2


def _table(v1: str, v2: str) -> dict | None:
    """A source table: its version-1 key (required there) or its version-2 key (None when not used)."""
    return CONFIG.get(v2, None) if V2 else CONFIG.get(v1)


def _path(table: dict | None) -> Path | None:
    return Path(table["file"]) if table and table.get("file") else None


SCENARIO = CONFIG.get("city.scenario")
DISTRICTS_FILE = CONFIG.path("sources.districts.file")
DISTRICT_ID_COLUMN = CONFIG.get("sources.districts.id_column")
DISTRICT_NAME_COLUMN = CONFIG.get("sources.districts.name_column")
DISTRICT_ID_WIDTH = CONFIG.get("sources.districts.id_width")
BUILDINGS = _table("sources.buildings", "buildings")
ALKIS_GPKG = _path(BUILDINGS)
ALKIS_LAYER = BUILDINGS["layer"] if BUILDINGS else None
CENSUS = _table("sources.census", "demand.census")
CENSUS_ZIP = _path(CENSUS)
CENSUS_CRS = CENSUS["crs"] if CENSUS else None
CENSUS_CELL_M = CENSUS["cell_size_m"] if CENSUS else None
# demand.py binds FORECAST["district_code_pattern"] as a default argument, so a v2 city without a long-layout
# population table gets the generic leading-digits pattern (used only by the long-table loader).
FORECAST = _table("sources.forecast", "population")
FORECAST_FILE = _path(FORECAST)
FORECAST = {"district_code_pattern": r"^(\d+)", **(FORECAST or {})} if V2 else FORECAST
SOCIAL = _table("sources.social", "vulnerability.four_indicator")
SOCIAL_FILE = _path(SOCIAL)

FORECAST_YEAR = FORECAST.get("reference_year") if V2 else FORECAST["year"]
FORECAST_AGE_GROUPS = dict(FORECAST.get("age_groups", {})) if V2 else dict(FORECAST["age_groups"])
SOCIAL_YEAR = SOCIAL["year"] if SOCIAL else None

# City-specific expectations, used only to validate the inputs ([validation]; None = not checked).
EXPECTED_DISTRICT_COUNT = CONFIG.get("validation.district_count", None)
EXPECTED_DISTRICT_CHILDREN_0_5 = CONFIG.get("validation.children_0_5_districts", None)

# Building-function codes accepted as residential, and the mixed-use code among them.
ALKIS_RESIDENTIAL_CODES = dict(BUILDINGS["residential_codes"]) if BUILDINGS else {}
ALKIS_MIXED_CODE = BUILDINGS["mixed_use_code"] if BUILDINGS else None

# The social indicators mapped onto the established index columns.
SOCIAL_INDICATORS = dict(SOCIAL["indicators"]) if SOCIAL else {}
if SOCIAL and set(SOCIAL_INDICATORS) != set(VULNERABILITY_COLUMNS):
    raise ConfigError(
        f"{CONFIG.source}: [sources.social.indicators] must map exactly the index columns "
        f"{VULNERABILITY_COLUMNS}, got {sorted(SOCIAL_INDICATORS)}"
    )

# Dormitory rule: an ALKIS residential building is excluded only if at least
# this share of its footprint lies inside OSM building=dormitory polygons that
# carry an explicit student signal. Smaller overlaps are reported as uncertain.
DORMITORIES = _table("sources.dormitories", "buildings.dormitories") or {}
DORM_EXCLUDE_MIN_SHARE = DORMITORIES.get("exclude_min_share")
DORM_REPORT_MIN_SHARE = DORMITORIES.get("report_min_share")
STUDENT_SIGNAL = re.compile(DORMITORIES["student_pattern"], re.I) if "student_pattern" in DORMITORIES else None
NON_STUDENT_SIGNAL = (
    re.compile(DORMITORIES["non_student_pattern"], re.I) if "non_student_pattern" in DORMITORIES else None
)

# Reviewed decisions for every residential building that may be a student-only residence (evidence: published
# residence listings, see the decisions file).
STUDENT_DECISIONS_FILE = Path(DORMITORIES["decisions_file"]) if not V2 or DORMITORIES.get("decisions_file") else None
STUDENT_DECISIONS_SHA256 = DORMITORIES.get("decisions_sha256", "")

if V2:
    FACILITIES = CONFIG.get("facilities")
    KITA_TAGS = dict(FACILITIES.get("osm_tags", {}))
    KITA_FILE = FACILITIES["file"] if FACILITIES["source"] == "file" else None
    BUILDING_TAGS_SOURCE = DORMITORIES.get("building_tags_source", "none")
else:
    KITA_TAGS = dict(CONFIG.get("osm.kita_tags"))
    # Optional offline sources: a local file of existing facilities instead of the OSM cache, and whether OSM
    # building tags (cache) are used to screen dormitories ("none": the reviewed decision file alone decides).
    KITA_FILE = CONFIG.get("osm.kita_file", None)
    BUILDING_TAGS_SOURCE = CONFIG.get("osm.building_tags_source", "cache")
if BUILDING_TAGS_SOURCE not in ("cache", "none"):
    raise ConfigError(f"{CONFIG.source}: building_tags_source must be 'cache' or 'none', got {BUILDING_TAGS_SOURCE!r}")

# Parameters of the district model ([model_parameters]; the other models' parameters are in core).
DISTRICT_TIME_LIMIT_SECONDS = PARAMETERS["district_time_limit_seconds"]  # default: 3,600 s
DISTRICT_MIP_GAP = PARAMETERS["district_mip_gap"]  # default: 1e-4

# Solver resources for the main runs. Objectives, time limits and MIP gaps stay
# as defined in the solver functions; only memory/thread behaviour changes.
RESOURCES = CONFIG.get("solver_resources")
NODEFILE_DIR = default_nodefile_dir(CONFIG)
NODEFILE_MIN_FREE_GB = float(RESOURCES["nodefile_min_free_gb"])
SOLVER_RESOURCES = {
    "Utilitarian": {},
    "Individual Rawlsian Stage 1": {},
    "Individual Rawlsian Stage 2": {
        "threads": RESOURCES["threads"],
        "nodefile_start": RESOURCES["nodefile_start"],
        "nodefile_dir": str(NODEFILE_DIR / "rawlsian_stage2"),
    },
    "District Rawlsian (screened)": {
        "threads": RESOURCES["threads"],
        "nodefile_start": RESOURCES["nodefile_start"],
        "nodefile_dir": str(NODEFILE_DIR / "district"),
    },
}

ALLOCATION_METHODS = ("footprint", "equal", "floor_area")
PRIMARY_ALLOCATION = "footprint"
