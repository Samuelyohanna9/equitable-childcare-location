"""Full workflows of a city: ``build``, ``solve`` and ``compare``.

``build`` makes the inputs from the raw data, ``solve`` runs the three models with Gurobi and ``compare`` checks
fresh results against reference results. Used by ``childcare-access build|solve|compare``.

Everything is written to a workspace (``[workspace] root`` of the configuration, default ``workspace/<config
name>``): the workspace may not be, or lie inside, ``data/``, ``results/``, a code or configuration folder or a
configured ``[outputs]`` folder, and existing workflow folders are never overwritten.

The build is offline. It reads the raw files of the configuration and the saved OSM data (network snapshot; Kitas
and building tags from the OSMnx cache or a local file) with all HTTP requests blocked, so the build never downloads
newer OSM data by itself; getting new source data is a separate step (``childcare_access.osm_download``,
used by the step-by-step notebook).

The modules used here read the ACTIVE configuration (``CHILDCARE_ACCESS_CONFIG``); the CLI selects it before
importing them.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, CityConfig, ConfigError, active_config
from .script_support import read_pickle

if TYPE_CHECKING:  # these modules read the active configuration when imported, so they are imported lazily
    from .all_demand_candidates import SparseCandidateRelations
    from .core import LocationModelResult


# workspace and logging
def workspace_root(override: str | Path | None = None) -> Path:
    """The workspace of the active configuration, checked to lie outside every protected folder."""
    config = active_config()
    root = Path(override) if override else Path(config.get("workspace.root", f"workspace/{config.source.stem}"))
    root = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
    protected = [
        PROJECT_ROOT / name
        for name in (
            "data",
            "results",
            "src",
            "tests",
            "config",
            "docs",
            "examples",
        )
    ]
    protected += [
        config.resolve(f"outputs.{k}")
        for k in ("final_inputs", "final_run", "audit")
        if config.get(f"outputs.{k}", None)
    ]
    for protected_folder in protected:
        folder = protected_folder.resolve()
        if root == folder or folder in root.parents or root in folder.parents:
            raise ConfigError(
                f"workspace {root} overlaps the protected folder {folder}; choose another [workspace] root"
            )
    if root == PROJECT_ROOT.resolve():
        raise ConfigError("the workspace cannot be the project root")
    return root


def _new_dir(path: Path) -> Path:
    """Create an empty output folder; refuse one that already holds files (nothing is overwritten)."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{path} already contains results; choose another name or workspace (nothing is overwritten)"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rel(path: Path) -> str:
    """Project-relative path where possible (reports carry no user-specific directories)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _abs(text: str) -> Path:
    """A configured path, relative to the project root unless it is absolute."""
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Checks:
    """Checks of a build: each is logged with the elapsed time; the first failure stops the build."""

    def __init__(self, log: Callable[[str], None]) -> None:
        self.items, self.log, self.started = [], log, time.perf_counter()

    def add(self, name: str, passed: bool, detail: object) -> None:
        """Record and log one check; raise if it failed."""
        self.items.append({"check": name, "passed": bool(passed), "detail": detail})
        self.log(
            f"[{time.perf_counter() - self.started:8.1f}s] {'PASS' if passed else 'FAIL'} {name}: {str(detail)[:240]}"
        )
        if not passed:
            raise RuntimeError(f"build check failed: {name}: {detail}")

    def note(self, message: str) -> None:
        """Log a message with the elapsed time."""
        self.log(f"[{time.perf_counter() - self.started:8.1f}s] {message}")


def _environment() -> dict:
    """Versions of Python, the platform and the main libraries, for the run records."""
    import geopandas
    import networkx
    import osmnx

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "geopandas": geopandas.__version__,
        "networkx": networkx.__version__,
        "osmnx": osmnx.__version__,
    }


# build
def _raw_files(config: CityConfig) -> list[tuple[str, str | None]]:
    """(config key, checksum key) of every raw file the build reads."""
    if config.version == 1:
        return [
            ("sources.districts.file", None),
            ("sources.forecast.file", None),
            ("sources.social.file", None),
            ("sources.buildings.file", "sources.buildings.sha256"),
            ("sources.census.file", "sources.census.sha256"),
            ("sources.dormitories.decisions_file", "sources.dormitories.decisions_sha256"),
            ("osm.network_snapshot", None),
        ]
    from .requirements import input_files

    files = [
        (f"{table}.{key}", f"{table}.sha256" if config.get(f"{table}.sha256", "") else None)
        for table, key in input_files(config.data)
    ]
    if config.get("buildings.dormitories.decisions_file", ""):
        files.append(("buildings.dormitories.decisions_file", "buildings.dormitories.decisions_sha256"))
    return files


def build(workspace: Path, log: Callable[[str], None] = print) -> Path:
    """Build the model inputs from the raw data into ``<workspace>/inputs`` (no solver).

    Version-1 configurations (Münster, the demo city) run these steps: districts and forecast; vulnerability index and
    Q4; residential buildings with the reviewed student-residence decisions; existing Kitas; pedestrian network from
    the snapshot (rule set of the configuration) and its layered routing graph; snapping; candidate pool (one
    candidate per network node of an eligible residential building); census-grid demand with fallback origins; demand
    locations; current access; sparse candidate relations; Q4 dominance screen. Version-2 configurations select the
    demand method (:mod:`demand_modes`); the common steps are in :mod:`assemble`.
    """
    from . import assemble, candidate_sources, demand_modes, vulnerability_modes
    from . import pedestrian_network as pn
    from . import settings as s
    from .core import (
        DISTRICT_NAME,
        LEAST_ADVANTAGED_QUANTILE,
        P_NEW_KITAS,
        download_kindergartens,
        get_bbox_wgs84,
    )
    from .demand import (
        apply_student_residence_decisions,
        classify_dormitories,
        load_alkis_residential_buildings,
        load_districts,
        load_forecast_children,
        load_osm_residential_tags,
    )
    from .network import load_kitas_from_file, osm_cache_only
    from .vulnerability import build_vulnerability

    config = s.CONFIG
    methods = config.methods
    mode, source = methods["demand"]["mode"], methods["candidates"]["source"]
    out = _new_dir(Path(workspace) / "inputs")
    checks = _Checks(log)
    started = time.perf_counter()

    # 1. Raw files and recorded checksums
    raw = {}
    for key, sha_key in _raw_files(config):
        path = config.resolve(key)
        if not path.is_file():
            raise FileNotFoundError(
                f"[{key}] {config.path(key)} not found; obtain the raw data first (see DATA_MANIFEST.md)"
            )
        raw[key] = {"path": config.path(key).as_posix(), "sha256": _sha256(path)}
        if sha_key and config.get(sha_key) and raw[key]["sha256"] != config.get(sha_key):
            raise RuntimeError(f"[{key}] checksum differs from the configuration")
    checks.add(
        "raw input files present (recorded checksums match)",
        True,
        {k: v["sha256"][:12] for k, v in raw.items()},
    )

    # 2. Districts, child totals, vulnerability
    districts = load_districts()
    population = config.get("population", None) if config.version == 2 else None
    if config.version == 1 or (population and population["layout"] == "long"):
        children, city = load_forecast_children()
        C = children.set_index("district_id")["children_0_5"]
        expected = s.EXPECTED_DISTRICT_CHILDREN_0_5
        checks.add(
            "forecast districts and total",
            expected is None or float(C.sum()) == float(expected),
            {
                "districts": len(children),
                "total": float(C.sum()),
                "citywide_row": city.get("children_0_5"),
            },
        )
    elif population:
        children = demand_modes.child_totals(
            {**population, "file": config.resolve("population.file")},
            districts,
            s.DISTRICT_ID_WIDTH,
        )
        checks.add(
            "district child totals",
            True,
            {
                "districts": len(children),
                "total": float(children["children_0_5"].sum()),
                "age_split": demand_modes.has_split(children),
            },
        )
    else:
        children = None
    if children is not None and config.version == 2:
        checks.add(
            "child totals metadata",
            True,
            {k: population[k] for k in ("count_type", "reference_year", "source")},
        )
    vmode, priority = methods["vulnerability"]["mode"], None
    if config.version == 1:
        districts, groups = build_vulnerability(districts)
        q4 = sorted(groups.loc[groups["least_advantaged"].astype(bool), DISTRICT_NAME])
        checks.add(
            "vulnerability index and Q4",
            len(q4) > 0,
            {"quantile": LEAST_ADVANTAGED_QUANTILE, "q4_districts": q4},
        )
    else:
        spec = {
            **config.get(f"vulnerability.{vmode}"),
            "file": config.resolve(f"vulnerability.{vmode}.file"),
        }
        if vmode == "four_indicator":
            districts, groups, priority = vulnerability_modes.four_indicator(districts, LEAST_ADVANTAGED_QUANTILE)
        elif vmode == "external_index":
            districts, groups, priority = vulnerability_modes.external_index(
                spec, districts, LEAST_ADVANTAGED_QUANTILE, s.DISTRICT_ID_WIDTH
            )
        else:
            districts, groups, priority = vulnerability_modes.priority_districts(spec, districts, s.DISTRICT_ID_WIDTH)
        checks.add(
            f"priority group: {priority['label']}",
            priority["selected_count"] > 0,
            {k: v for k, v in priority.items() if k not in ("label", "selected_ids")},
        )

    # 3. Residential buildings and student-residence decisions (when the methods use buildings)
    direct = config.get("demand.direct", {}) if mode == "direct" else {}
    needs_buildings = (
        mode != "direct"
        or methods["candidates"]["source"] == "residential_buildings"
        or (direct.get("geometry") == "polygons" and direct.get("polygon_allocation") == "building_area")
    )
    buildings = None
    if needs_buildings:
        buildings, alkis_audit = load_alkis_residential_buildings(districts)
        if s.BUILDING_TAGS_SOURCE == "cache":
            osm = load_osm_residential_tags(districts)
            buildings, matches = classify_dormitories(buildings, osm)
        else:
            buildings["exclude_student_dormitory"] = False
            matches = pd.DataFrame(columns=["building_id", "decision"])
        if s.STUDENT_DECISIONS_FILE is not None:
            buildings, _ = apply_student_residence_decisions(buildings, matches)
        checks.add(
            "residential buildings and reviewed exclusions",
            alkis_audit["duplicate_oid"] == 0 and alkis_audit["duplicate_geometry"] == 0,
            {
                "buildings": len(buildings),
                "excluded": int(buildings["exclude_student_dormitory"].sum()),
                "building_tags": s.BUILDING_TAGS_SOURCE,
            },
        )

    # 4. Existing facilities (offline: local file, or the OSMnx cache with HTTP blocked)
    if s.KITA_FILE:
        kita_key = "osm.kita_file" if config.version == 1 else "facilities.file"
        kitas = load_kitas_from_file(config.resolve(kita_key))
        kita_source = config.get(kita_key)
    else:
        with osm_cache_only():
            kitas = download_kindergartens(get_bbox_wgs84(districts), tags=s.KITA_TAGS)
        kita_source = f"OSMnx cache ({config.get('osm.cache_dir')}), tags {s.KITA_TAGS}"
    checks.add("existing facilities", len(kitas) > 0, {"facilities": len(kitas), "source": kita_source})

    # 5-6. Pedestrian network from the snapshot, layered routing graph; snapping (common assembly)
    rule_set = config.get("osm.network_rule_set")
    graph, layered = assemble.walking_network(districts, rule_set, config.resolve("osm.network_snapshot"), out, checks)
    buildings, kitas, snapping = assemble.snap(graph, buildings, kitas, checks)

    # 7. Candidates of the selected source (demand-node candidates follow the demand locations, step 8)
    sites = None
    if source == "residential_buildings":
        # one candidate per network node of an eligible residential building; only the node set matters, so without
        # district totals (direct demand) every district with buildings gets weight 1
        basis = (
            children
            if children is not None
            else pd.DataFrame(
                {
                    "district_id": sorted(
                        buildings.loc[~buildings["exclude_student_dormitory"].astype(bool), "district_id"].unique()
                    ),
                    "children_0_5": 1.0,
                }
            )
        )
        candidates = candidate_sources.residential(graph, buildings, basis, snapping, districts, groups)
        eligible = set(buildings.loc[~buildings["exclude_student_dormitory"].astype(bool), "nearest_node"])
        checks.add(
            "candidates: one per node of an eligible residential building",
            set(candidates["base_node"]) == eligible and candidates["base_node"].is_unique,
            {"candidates": len(candidates)},
        )
    elif source == "file":
        candidates, sites = candidate_sources.supplied(
            {**config.get("candidates"), "file": config.resolve("candidates.file")},
            districts,
            graph,
        )
        excluded = sites[sites["status"] == "excluded"]
        checks.add(
            "candidate sites (supplied; eligibility as given by the source)",
            True,
            {
                "supplied": len(sites),
                "candidates": len(candidates),
                "excluded": len(excluded),
                "excluded_by_reason": excluded["reason"]
                .str.replace(r"site .*$", "site ...", regex=True)
                .value_counts()
                .to_dict(),
                "on_district_boundary": int(sites["on_boundary"].sum()),
                "max_snap_m": round(float(sites.loc[sites["status"] == "candidate", "snap_distance_m"].max()), 1),
                "table": "candidate_sites.csv",
            },
        )

    # 8. Demand origins of the selected method, demand locations (common assembly)
    building_demand, variant = (
        None,
        {"census_hybrid": "census_grid", "building_area": "footprint", "direct": "direct"}[mode],
    )
    if mode == "census_hybrid":
        origins, building_demand, fallback, _ = demand_modes.census_hybrid(buildings, children, districts, s.CENSUS_CRS)
        fallback.to_csv(out / "fallback_census_cells.csv", index=False)
        detail = {"fallback_origins": len(fallback)}
    elif mode == "building_area":
        origins, building_demand = demand_modes.building_area(buildings, children)
        detail = {}
    else:
        origins, detail = demand_modes.direct(
            {**direct, "file": config.resolve("demand.direct.file")}, districts, graph, buildings
        )
    if children is not None and mode != "direct":
        totals = children.set_index("district_id")["children_0_5"]
    else:
        totals = origins.groupby("district_id")["children_0_5"].sum().reindex(districts["district_id"], fill_value=0.0)
        if children is not None:  # direct demand with district totals: compared per district, never rescaled
            policy, tolerance = (
                direct.get("consistency", "report"),
                float(direct.get("consistency_tolerance", 0.0)),
            )
            table = demand_modes.consistency(totals, children, tolerance)
            table.to_csv(out / "direct_vs_district_totals.csv", index=False)
            flagged = table[table["status"] != "match"]
            by_status = {
                s: sorted(table.loc[table["status"] == s, "district_id"])
                for s in ("differs", "no_direct_demand", "no_district_total")
            }
            checks.add(
                f"direct demand vs [population] district totals (policy '{policy}', tolerance {tolerance})",
                policy == "report" or flagged.empty,
                {
                    "direct_total": float(table["direct_total"].sum()),
                    "population_total": float(table["population_total"].sum()),
                    "max_abs_difference": float(table["abs_difference"].max()),
                    **{s: d for s, d in by_status.items() if d},
                    "table": "direct_vs_district_totals.csv",
                },
            )
            if not flagged.empty:
                checks.note(
                    f"NOTE: {len(flagged)} district(s) differ from the supplied totals or lack coverage "
                    f"(policy 'report': kept as supplied, not rescaled): {by_status}"
                )
        sums = ["children_0_5"] + [
            a for a in demand_modes.AGES if a in origins
        ]  # district totals of the supplied demand
        children = (
            origins.groupby("district_id")[sums]
            .sum()
            .reindex(districts["district_id"], fill_value=0.0)
            .rename_axis("district_id")
            .reset_index()
        )
    demand = assemble.demand_locations(
        graph, origins, districts, groups, totals, demand_modes.describe(methods), checks, detail
    )
    if priority is not None:  # no priority-group demand at all stops the build; districts without demand are reported
        positive = demand.loc[demand["least_advantaged"] & (demand["estimated_demand"] > 0)]
        located = set(positive[DISTRICT_NAME])
        empty = sorted(set(priority["selected"]) - located)
        priority["districts_with_demand"] = sorted(located)
        priority["districts_without_demand"] = empty  # selected, but outside every district mean (zero demand)
        checks.add(
            "priority-group demand locations",
            bool(located),
            {
                "locations": len(positive),
                "children": float(positive["estimated_demand"].sum()),
                "selected districts": len(priority["selected"]),
                "with positive demand": len(located),
                "excluded from district means (zero demand)": empty,
                "note": "district classification: every location inherits its district's group; the district "
                "Rawlsian maximum runs over the priority districts with positive demand",
            },
        )
        if empty:
            checks.note(
                f"NOTE: priority district(s) without any demand: {empty}; selected by the vulnerability "
                "definition, but not part of any district mean or of the district Rawlsian maximum"
            )

    if source == "demand_nodes":  # EXPERIMENTAL: the demand-location nodes as candidates
        candidates = candidate_sources.demand_nodes(demand)
        checks.add(
            "candidates: demand-location nodes (EXPERIMENTAL modelling assumption, not developable sites)",
            True,
            {"candidates": len(candidates)},
        )
    enough = candidate_sources.check_enough(candidates, P_NEW_KITAS, source)
    if config.version == 2:
        checks.add("enough distinct candidate nodes for P", True, enough)

    # 9-10. Current access, sparse relations, Q4 dominance screen, model sizes, feasibility (common assembly)
    current, _, screen = assemble.relations_and_screen(layered, demand, kitas, candidates, P_NEW_KITAS, out, checks)

    # 11. Save (the same format as the Münster model inputs)
    buildings_out = None
    if buildings is not None:
        buildings_out = buildings.copy()
        if building_demand is not None:
            buildings_out[f"children_0_5_{variant}"] = building_demand.to_numpy()
    bundle = {
        "scenario": s.SCENARIO,
        "districts": districts,
        "district_groups": groups,
        "children": children,
        "candidates": candidates,
        "kitas": kitas,
        "demand": demand,
        "demand_by_method": {variant: demand},
        "current_access": current,
        "screen_retained_indices": screen.retained_indices,
        "building_demand_column": f"children_0_5_{variant}" if building_demand is not None else None,
        "allocation_variant": variant,
        "excluded_census_cells": [],
        "network": {
            "version": rule_set,
            "source": config.get("osm.network_snapshot"),
            "base_graph": "walk_graph_base.pkl",
            "routing_graph": "walk_graph_layered.pkl",
            "facility_node_offset": pn.LAYER_C,
        },
    }
    report = {
        "scenario": s.SCENARIO,
        "built_by": "childcare-access build",
        "config": config.source.name,
        "config_sha256": _sha256(config.source),
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment(),
        "raw_inputs": raw,
        "kita_source": kita_source,
        "indicator_labels": s.SOCIAL_INDICATORS,
        "indicator_definitions": (s.SOCIAL or {}).get("indicator_definitions", {}),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "all_checks_passed": True,
        "checks": checks.items,
    }
    if config.version == 2:
        report["methods"] = methods
        report["priority_group"] = priority
        bundle["priority_group"] = priority
        origins.to_csv(out / "demand_origins.csv", index=False)
        vulnerability_modes.standard_table(districts).to_csv(out / "priority_groups.csv", index=False)
        if sites is None:  # residential or demand-node candidates: one row per candidate
            sites = pd.DataFrame(
                {
                    "site_id": candidates["facility_id"],
                    "base_node": candidates["base_node"],
                    "source": candidates["candidate_source"],
                    "status": "candidate",
                    "reason": "",
                }
            )
        sites.to_csv(out / "candidate_sites.csv", index=False)
        report["candidates"] = {
            "source": source,
            "count": len(candidates),
            "experimental": source == "demand_nodes",
            "eligibility": {
                "file": "as given by the source",
                "demand_nodes": "not assessed (modelling assumption)",
                "residential_buildings": "eligible residential buildings (default rule)",
            }[source],
        }
    if config.version == 2:  # no secret-like value reaches a version-2 report
        from . import provenance

        report = provenance.redact(report)
    assemble.save(out, bundle, buildings_out, children, report)
    if config.version == 2:  # machine-readable provenance (after saving: it fingerprints the saved inputs)
        candidate_record = dict(report["candidates"])
        if sites is not None and source == "file":
            candidate_record["policies"] = {
                k: config.get(f"candidates.{k}", None)
                for k in (
                    "eligible_column",
                    "eligible_values",
                    "max_snap_m",
                    "shared_node",
                    "boundary_site",
                    "polygon_point",
                )
            }
            candidate_record["supplied"] = len(sites)
            candidate_record["excluded"] = sites.loc[sites["status"] == "excluded", ["site_id", "reason"]].to_dict(
                "records"
            )
            candidate_record["boundary_assignments"] = sites.loc[sites["boundary_assignment"] != "", "site_id"].tolist()
        record = provenance.build_record(
            config,
            raw,
            priority,
            candidate_record,
            origins,
            graph,
            rule_set,
            report["environment"],
            demand_modes.describe(methods),
            out,
        )
        provenance.write(out, record)
        checks.note(
            f"provenance: {out / 'provenance.json'} (scientific identity: {len(record['identity']['scientific'])} "
            "components)"
        )
    checks.note(f"build complete: {out}")
    return out


# solve
class _RecordingGurobi:
    """``gurobipy`` stand-in whose Model() records each model and gives it its own log file.

    Every other attribute is taken from ``gurobipy``.
    """

    def __init__(self, gurobipy: ModuleType, directory: Path, prefix: str, overrides: dict | None = None) -> None:
        self.gurobipy, self.directory, self.prefix, self.models, self.overrides = (
            gurobipy,
            directory,
            prefix,
            [],
            overrides or {},
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gurobipy, name)

    def Model(self, name: str = "", *args: Any, **kwargs: Any) -> Any:  # noqa: N802 (gurobipy API)
        """Create a model that logs to its own file; apply the test overrides of its name."""
        model = self.gurobipy.Model(name, *args, **kwargs)
        log_path = self.directory / f"{self.prefix}_{len(self.models) + 1:02d}_{name}.log"
        model.Params.LogFile = str(log_path)
        model.Params.LogToConsole = 0
        self.models.append((model, log_path))
        return _OverrideAtOptimize(model, self.overrides[name]) if name in self.overrides else model


class _OverrideAtOptimize:
    """Test hook: applies parameters immediately before optimize(), after the solver set its own (e.g. TimeLimit)."""

    def __init__(self, model: Any, params: dict[str, Any]) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_params", params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._model, name, value)

    def optimize(self, *args: Any, **kwargs: Any) -> None:
        """Apply the override parameters, then optimise (optionally stop at the first incumbent)."""
        params = dict(self._params)
        stop_at_first = params.pop("stop_at_first_solution", False)
        for key, value in params.items():
            self._model.setParam(key, value)
        if not stop_at_first:
            return self._model.optimize(*args, **kwargs)

        def callback(
            model: Any, where: int
        ) -> None:  # time limit reached as soon as the first incumbent (e.g. the MIP start) exists
            from gurobipy import GRB

            if where == GRB.Callback.MIPSOL:
                model.cbSetParam("TimeLimit", 0.0)

        return self._model.optimize(callback)


def _model_record(
    gurobipy: ModuleType, model: Any, log_path: Path, role: str, directory: Path, candidate_ids: list[str]
) -> dict:
    """Solver record of one solved model; writes its incumbent to ``<log stem>.sol``."""
    from .all_demand_candidates import _gurobi_status_name

    record = {
        "role": role,
        "model_name": model.ModelName,
        "status": _gurobi_status_name(model.Status, gurobipy.GRB),
        "solution_count": int(model.SolCount),
        "runtime_seconds": float(model.Runtime),
        "node_count": float(model.NodeCount),
        "rows": int(model.NumConstrs),
        "columns": int(model.NumVars),
        "nonzeros": int(model.NumNZs),
        "time_limit_seconds": float(model.Params.TimeLimit),
        "mip_gap_tolerance": float(model.Params.MIPGap),
        "mip_focus": int(model.Params.MIPFocus),
        "threads_param": int(model.Params.Threads),
        "seed": int(model.Params.Seed),
        "nodefile_start_param": float(model.Params.NodefileStart),
        "log_file": log_path.name,
        "incumbent_objective": None,
        "best_bound": None,
        "mip_gap": None,
        "selected_candidate_ids": "",
    }
    if model.SolCount > 0:
        record.update(
            {
                "incumbent_objective": float(model.ObjVal),
                "best_bound": float(model.ObjBound),
                "mip_gap": float(model.MIPGap),
            }
        )
        opened = [v for v in model.getVars() if v.VarName.startswith("open_candidate[") and v.X > 0.5]
        record["selected_candidate_ids"] = " | ".join(
            candidate_ids[int(v.VarName[len("open_candidate[") : -1])] for v in opened
        )
        model.write(str(directory / f"{log_path.stem}.sol"))
    return record


def _save(
    name: str, result: LocationModelResult, records: list[dict], validation: pd.DataFrame, directory: Path
) -> None:
    """Write the assignment, validation, certified solver records, JSON summary and pickle of one model."""
    from .solutions import certify, certify_stage

    certification = certify(records)
    for r in records:
        r["certification"] = certify_stage(r)
    result.assignment.to_csv(directory / f"{name}_assignment.csv", index=False)
    validation.to_csv(directory / f"{name}_independent_validation.csv", index=False)
    pd.DataFrame(records).to_csv(directory / f"{name}_solver_records.csv", index=False)
    details = {k: v for k, v in result.solver_details.items() if k != "gap_trace"}
    summary = {
        "scenario": result.name,
        "status": result.status,
        "certification": certification,
        "objective_value": result.objective_value,
        "secondary_objective_value": result.secondary_objective_value,
        "selected_candidate_ids": result.selected_candidate_ids,
        "solver_details": details,
        "models": records,
    }
    (directory / f"{name}_result.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    with (directory / f"{name}_result.pkl").open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


LARGE_MODEL_PAIRS = 100_000  # above this many candidate-location pairs a solve needs explicit confirmation


def solve(
    workspace: Path,
    inputs: Path | None = None,
    run_name: str | None = None,
    p: int | None = None,
    models: Sequence[str] = ("utilitarian", "rawlsian", "district"),
    district_warm_start: str = "same-run",
    solver_overrides: dict | None = None,
    confirm_large: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    """Solve the three models with Gurobi on built inputs; results go to ``<workspace>/runs/<run_name>``.

    Time limits, MIP gaps, MIPFocus, Stage 2 tolerances and resources come from the configuration's
    ``[model_parameters]``/``[solver_resources]``.
    ``district_warm_start``: ``"same-run"`` (default: the better of this run's Utilitarian and
    Individual sites, mapped to retained dominators, used only as a MIP start) or ``"none"``. Saved reference
    solutions are never read. ``solver_overrides`` (model name -> Gurobi parameters) exists for tests only.
    """
    from . import all_demand_candidates as adc
    from . import settings as s
    from . import validation as v
    from .core import (
        GUROBI_MIP_GAP,
        GUROBI_TIME_LIMIT_SECONDS,
        P_NEW_KITAS,
        RAWLS_STAGE1_MIP_GAP,
        RAWLS_STAGE2_MIP_GAP,
        RAWLS_STAGE2_TOLERANCES,
    )
    from .district_demand_maximin import (
        solve_sparse_district_demand_maximin,
        validate_district_demand_maximin_solution,
    )
    from .optimisation import nodefile_preflight, screened_relations

    if district_warm_start not in ("same-run", "none"):
        raise ValueError("district_warm_start must be 'same-run' or 'none'")
    P = int(p if p is not None else P_NEW_KITAS)
    inputs = Path(inputs) if inputs else Path(workspace) / "inputs"
    report = json.loads((inputs / "validation_report.json").read_text(encoding="utf-8"))
    if not report["all_checks_passed"]:
        raise RuntimeError(f"{inputs}: input validation did not pass; refusing to solve")
    out = Path(workspace) / "runs" / (run_name or f"P{P}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already contains results; choose another --run-name (nothing is overwritten)")
    with (inputs / "inputs_bundle.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    with (inputs / "sparse_relations_all_demand.pkl").open("rb") as handle:
        relations = pickle.load(handle)
    with (inputs / bundle["network"]["routing_graph"]).open("rb") as handle:
        graph = pickle.load(handle)
    demand, current, candidates = bundle["demand"], bundle["current_access"], bundle["candidates"]
    v.check_optimisation_feasibility(
        P, candidates, current, relations, screened_count=len(bundle["screen_retained_indices"])
    )
    if relations.pair_count > LARGE_MODEL_PAIRS and not confirm_large:
        raise RuntimeError(
            f"large model: {relations.pair_count:,} candidate-location pairs, {len(candidates):,} candidates. "
            "A run of this size can take hours and needs a full Gurobi licence; repeat with --yes to start it."
        )
    try:
        import gurobipy
    except ImportError as error:
        raise ImportError(
            "solve needs gurobipy and a Gurobi licence (uv sync installs gurobipy; verification does not need it)"
        ) from error
    _new_dir(out)
    started = time.perf_counter()

    def note(message: str) -> None:
        log(f"[{time.perf_counter() - started:8.1f}s] {message}")

    manifest = {
        "scenario": s.SCENARIO,
        "config": s.CONFIG.source.name,
        "config_sha256": _sha256(s.CONFIG.source),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": _rel(inputs),
        "inputs_bundle_sha256": _sha256(inputs / "inputs_bundle.pkl"),
        "relations_sha256": _sha256(inputs / "sparse_relations_all_demand.pkl"),
        "gurobi_version": ".".join(map(str, gurobipy.gurobi.version())),
        "facility_budget": P,
        "models": list(models),
        "district_warm_start": district_warm_start,
        "reference_solutions_read": False,
        "settings": {
            "Utilitarian": {
                "TimeLimit": GUROBI_TIME_LIMIT_SECONDS,
                "MIPGap": GUROBI_MIP_GAP,
                "Threads": "default",
                "Seed": "default (0)",
            },
            "Individual Rawlsian Stage 1": {
                "TimeLimit": GUROBI_TIME_LIMIT_SECONDS,
                "MIPGap": RAWLS_STAGE1_MIP_GAP,
                "MIPFocus": 1,
                "Threads": "default",
                "Seed": "default (0)",
            },
            "Individual Rawlsian Stage 2": {
                "TimeLimit": GUROBI_TIME_LIMIT_SECONDS,
                "MIPGap": RAWLS_STAGE2_MIP_GAP,
                "MIPFocus": 1,
                "bound_tolerances": RAWLS_STAGE2_TOLERANCES,
                "Seed": "default (0)",
                **s.SOLVER_RESOURCES["Individual Rawlsian Stage 2"],
            },
            "District Rawlsian (screened)": {
                "TimeLimit": s.DISTRICT_TIME_LIMIT_SECONDS,
                "MIPGap": s.DISTRICT_MIP_GAP,
                "MIPFocus": 1,
                "Seed": "default (0)",
                **s.SOLVER_RESOURCES["District Rawlsian (screened)"],
            },
        },
        "nodefile_preflight": nodefile_preflight() if "rawlsian" in models or "district" in models else None,
        "environment": _environment(),
    }
    (out / "run_manifest_solve.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    def recorded(prefix: str, directory: Path) -> tuple[_RecordingGurobi, tuple[Any, int]]:
        recorder = _RecordingGurobi(gurobipy, directory, prefix, solver_overrides)
        saved = (adc._require_gurobi, adc.GUROBI_OUTPUT_FLAG)
        adc._require_gurobi, adc.GUROBI_OUTPUT_FLAG = (lambda: (recorder, gurobipy.GRB)), 1
        return recorder, saved

    def restore(saved: tuple[Any, int]) -> None:
        adc._require_gurobi, adc.GUROBI_OUTPUT_FLAG = saved

    solved = {}
    if "utilitarian" in models:
        directory = out / "utilitarian"
        directory.mkdir()
        note("Utilitarian ...")
        recorder, saved = recorded("utilitarian", directory)
        try:
            result = adc.solve_sparse_utilitarian(demand, current, relations, p_new_kitas=P)
        finally:
            restore(saved)
        records = [
            _model_record(gurobipy, m, lp, "Utilitarian", directory, relations.candidate_ids)
            for m, lp in recorder.models
        ]
        validation = adc.validate_selected_solution_with_dijkstra(graph, demand, current, candidates, result)
        _save("utilitarian", result, records, validation, directory)
        solved["utilitarian"] = result.selected_candidate_ids
        note(
            f"Utilitarian: {records[-1]['status']}, objective {result.objective_value:.6f}, gap "
            f"{records[-1]['mip_gap']}"
        )

    if "rawlsian" in models:
        directory = out / "rawlsian"
        directory.mkdir()
        resources = s.SOLVER_RESOURCES["Individual Rawlsian Stage 2"]
        note("Individual Rawlsian Stage 1 + lexicographic Stage 2 ...")
        recorder, saved = recorded("rawlsian", directory)
        try:
            result = adc.solve_sparse_rawlsian(
                demand,
                current,
                relations,
                p_new_kitas=P,
                stage2_threads=resources["threads"],
                stage2_nodefile_start=resources["nodefile_start"],
                stage2_nodefile_dir=resources["nodefile_dir"],
            )
        finally:
            restore(saved)
        records = [
            _model_record(
                gurobipy,
                m,
                lp,
                "Individual Rawlsian Stage 1" if i == 0 else f"Individual Rawlsian Stage 2 (attempt {i})",
                directory,
                relations.candidate_ids,
            )
            for i, (m, lp) in enumerate(recorder.models)
        ]
        validation = adc.validate_selected_solution_with_dijkstra(graph, demand, current, candidates, result)
        _save(
            "rawlsian",
            result,
            [records[0], records[-1]] if len(records) > 1 else records,
            validation,
            directory,
        )
        if len(records) > 2:  # earlier Stage 2 attempts (infeasible with a tighter bound) are saved as well
            pd.DataFrame(records).to_csv(directory / "rawlsian_all_attempts.csv", index=False)
        solved["rawlsian"] = result.selected_candidate_ids
        note("Individual Rawlsian: " + "; ".join(f"{r['role']}: {r['status']} gap {r['mip_gap']}" for r in records))

    if "district" in models:
        directory = out / "district"
        directory.mkdir()
        resources = s.SOLVER_RESOURCES["District Rawlsian (screened)"]
        screened = screened_relations(relations, bundle["screen_retained_indices"])
        warm_nodes, warm_note = (None, "none (option 'none')")
        if district_warm_start == "same-run":
            warm_nodes, warm_note = _district_warm_start(
                demand, current, relations, screened, inputs, solved, P, directory
            )
        note(f"District Rawlsian on {len(screened.candidate_ids)} screened candidates; warm start: {warm_note}")
        from . import district_demand_maximin as ddm

        saved_ddm = ddm._require_gurobi
        if solver_overrides and "q4_district_estimated_demand_maximin" in solver_overrides:  # test hook only
            proxy = _RecordingGurobi(gurobipy, directory, "district_test_hook", solver_overrides)
            ddm._require_gurobi = lambda: (proxy, gurobipy.GRB)
        try:
            result = solve_sparse_district_demand_maximin(
                demand,
                current,
                screened,
                p_new_kitas=P,
                time_limit_seconds=s.DISTRICT_TIME_LIMIT_SECONDS,
                mip_gap=s.DISTRICT_MIP_GAP,
                mip_focus=1,
                threads=resources["threads"],
                nodefile_start=resources["nodefile_start"],
                nodefile_dir=resources["nodefile_dir"],
                warm_start_candidate_nodes=warm_nodes,
                model_artifact_dir=directory,
                model_artifact_stem=f"district_rawlsian_p{P}",
            )
        finally:
            ddm._require_gurobi = saved_ddm
        d = result.solver_details
        records = [
            {
                "role": "District Rawlsian (screened)",
                "model_name": "q4_district_estimated_demand_maximin",
                "status": d["status"],
                "solution_count": None,
                "runtime_seconds": d["gurobi_runtime_seconds"],
                "node_count": d["node_count"],
                "rows": d["num_constraints"],
                "columns": d["num_variables"],
                "nonzeros": d["num_nonzeros"],
                "time_limit_seconds": d["time_limit_seconds"],
                "mip_gap_tolerance": d["mip_gap_tolerance"],
                "mip_focus": d["mip_focus"],
                "threads_param": d["threads"],
                "seed": "default (0)",
                "nodefile_start_param": d["nodefile_start"],
                "incumbent_objective": d["incumbent_objective"],
                "best_bound": d["best_bound"],
                "mip_gap": d["mip_gap"],
                "selected_candidate_ids": " | ".join(result.selected_candidate_ids),
                "warm_start": warm_note,
            }
        ]
        validation, district_compare = validate_district_demand_maximin_solution(
            result, graph, demand, current, candidates, p_new_kitas=P
        )
        district_compare.to_csv(directory / "district_independent_comparison.csv", index=False)
        _save("district", result, records, validation, directory)
        note(
            f"District: {d['status']}, incumbent {d['incumbent_objective']:.6f}, bound {d['best_bound']:.6f}, gap "
            f"{d['mip_gap']}"
        )

    note(f"solve complete: {out}")
    return out


def _district_warm_start(
    demand: pd.DataFrame,
    current: pd.DataFrame,
    relations: SparseCandidateRelations,
    screened: SparseCandidateRelations,
    inputs: Path,
    solved: dict,
    P: int,
    directory: Path,
) -> tuple[list[int] | None, str]:
    """MIP start for the district model from THIS run's Utilitarian/Individual sites (the default).

    Each set is evaluated on the district objective and the best is mapped to retained candidates through the
    validated dominance chains (a dominator never worsens any Q4 walking time). Only a start; it does not constrain
    the model.

    Returns:
        The network nodes of the start (None if no solved set is usable) and a note for the log.
    """
    from .all_demand_candidates import evaluate_selected_sparse_solution
    from .district_demand_maximin import q4_district_demand_adjusted_accessibility

    try:
        mapping = pd.read_csv(inputs / "dominance_mapping.csv")
    except pd.errors.EmptyDataError:  # the screen removed no candidate: nothing to map
        mapping = pd.DataFrame(columns=["removed_facility_id", "retained_dominator_facility_id"])
    to_retained = dict(zip(mapping["removed_facility_id"], mapping["retained_dominator_facility_id"], strict=True))
    node_of = dict(zip(relations.candidate_ids, relations.candidate_nodes, strict=True))
    index_of = {cid: i for i, cid in enumerate(relations.candidate_ids)}
    options = []
    for name in ["utilitarian", "rawlsian"]:
        if name not in solved:
            continue
        mapped = [to_retained.get(cid, cid) for cid in solved[name]]
        if len({node_of[c] for c in mapped}) != P:
            continue
        assignment = evaluate_selected_sparse_solution(demand, current, relations, [index_of[c] for c in mapped], None)
        value = float(
            q4_district_demand_adjusted_accessibility(assignment)["estimated_demand_adjusted_mean_walk_time"].max()
        )
        options.append((value, name, mapped))
    if not options:
        return None, "none (no solution of this run available)"
    value, name, mapped = min(options)
    (directory / "warm_start.json").write_text(
        json.dumps(
            {
                "source": f"this run's {name} solution",
                "candidate_ids": mapped,
                "district_objective": value,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [
        node_of[c] for c in mapped
    ], f"this run's {name} solution mapped to retained dominators; district objective {value:.6f}"


# compare
def compare_inputs(
    workspace: Path,
    inputs: Path | None = None,
    reference_inputs: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Compare freshly built inputs with the reference inputs; writes ``<workspace>/compare/inputs``."""
    from . import settings as s

    inputs = Path(inputs) if inputs else Path(workspace) / "inputs"
    reference_inputs = Path(reference_inputs) if reference_inputs else s.CONFIG.resolve("outputs.final_inputs")
    out = _new_dir(Path(workspace) / "compare" / "inputs")
    report = {
        "inputs": _rel(inputs),
        "reference_inputs": _rel(reference_inputs),
        **_compare_inputs(inputs, reference_inputs),
    }
    report["consistent"] = bool(report.get("identical_model_inputs"))
    (out / "compare_inputs_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for key, value in report.items():
        log(f"{key}: {value}")
    log(("IDENTICAL MODEL INPUTS" if report["consistent"] else "INPUTS DIFFER") + f" ({out})")
    return report


def compare(
    workspace: Path,
    run: Path | None = None,
    reference_inputs: Path | None = None,
    reference_run: Path | None = None,
    tolerance: float = 1e-6,
    log: Callable[[str], None] = print,
) -> dict:
    """Compare a fresh run (and its inputs) with a reference (default: the ``[outputs]`` results of the configuration).

    Per model: the fresh sites are evaluated on the REFERENCE inputs by Dijkstra, so both solutions are judged by the
    same data. Outcome labels:
    - ``identical sites``: the same facility set;
    - ``alternative optimum``: different sites, same objective value (within ``tolerance``, relative) - allowed when
      optima are not unique;
    - ``feasible, worse`` / ``feasible, better``: a valid solution with a different objective. ``better`` against a
      proven-optimal reference is an inconsistency (it would contradict the proof);
    plus the certification of both runs. The Individual Rawlsian model is judged on Stage 1 (Z*) and, separately, on
    the Stage 2 objective; the exact Stage 2 optimum by enumeration (if saved) is a lower bound for any Stage 2 value.
    Writes ``<workspace>/compare/<run name>/compare_report.json`` and ``.md``.
    """
    from . import settings as s
    from . import solutions as sol
    from .evaluation import (
        accessibility_indicators,
        district_means,
        lexicographic_stage2_objective,
        scenario_times,
    )
    from .saved_inputs import load_primary_inputs
    from .validation import check_solution

    run = Path(run) if run else Path(workspace) / "runs" / f"P{s.CONFIG.parameters['facility_budget']}"
    reference_inputs = Path(reference_inputs) if reference_inputs else s.CONFIG.resolve("outputs.final_inputs")
    reference_run = Path(reference_run) if reference_run else s.CONFIG.resolve("outputs.final_run")
    out = _new_dir(Path(workspace) / "compare" / run.name)
    reference = sol.load_reported_solutions(
        run_dir=reference_run,
        config=s.CONFIG if reference_run == s.CONFIG.resolve("outputs.final_run") else _NoReported(),
    )
    fresh = sol.load_reported_solutions(run_dir=run, config=_NoReported(), require_all=False)
    alternatives = sol.load_alternative_solutions(run_dir=reference_run)
    data = load_primary_inputs(reference_inputs)
    demand = data["demand"].set_index("demand_id")
    nodes = data["candidate_sites"].set_index("facility_id")["nearest_node"]
    kitas = data["existing_kitas"]["nearest_node"]
    P = len(next(iter(reference.values())).sites)
    report = {
        "run": _rel(run),
        "reference_run": _rel(reference_run),
        "reference_inputs": _rel(reference_inputs),
        "tolerance": tolerance,
        "inputs": _compare_inputs(
            _abs(json.loads((run / "run_manifest_solve.json").read_text(encoding="utf-8"))["inputs"]),
            reference_inputs,
        ),
        "models": {},
        "inconsistencies": [],
    }

    def objectives(key: str, sites: Sequence[str]) -> dict:
        t = scenario_times(data["graph"], demand, kitas, nodes, sites)
        q4 = demand["least_advantaged"].astype(bool)
        if key == "utilitarian":
            return {
                "objective": float((t * demand["estimated_demand"]).sum()),
                "indicators": accessibility_indicators(t, demand),
            }
        if key == "individual":
            return {
                "objective": float(t[q4].max()),
                "stage2_objective": lexicographic_stage2_objective(t, demand),
                "indicators": accessibility_indicators(t, demand),
            }
        return {
            "objective": float(district_means(t, demand).max()),
            "indicators": accessibility_indicators(t, demand),
        }

    def relation(a: float, b: float) -> str:
        if abs(a - b) <= tolerance * max(1.0, abs(b)):
            return "equal"
        return "worse" if a > b else "better"

    for key, ref in reference.items():
        new = fresh.get(key)
        entry = {"reference_sites": list(ref.sites), "reference_certification": ref.certification}
        if new is None:
            entry["outcome"] = "not solved in this run"
            report["models"][key] = entry
            continue
        if len(new.sites) != P:
            entry.update(
                outcome=f"not comparable: facility budget {len(new.sites)} vs reference {P}",
                fresh_sites=list(new.sites),
            )
            report["models"][key] = entry
            continue
        unknown = [x for x in new.sites if x not in nodes.index]
        if unknown:
            entry.update(outcome="sites not in the reference candidate set", unknown_sites=unknown)
            report["inconsistencies"].append(f"{key}: fresh sites {unknown} are not reference candidates")
            report["models"][key] = entry
            continue
        check_solution(new.sites, P, data["candidate_sites"])
        a, b = objectives(key, new.sites), objectives(key, ref.sites)
        rel = relation(a["objective"], b["objective"])
        same = sorted(new.sites) == sorted(ref.sites)
        outcome = (
            "identical sites"
            if same
            else (
                "alternative optimum (same objective, other sites)" if rel == "equal" else f"feasible, {rel} objective"
            )
        )
        entry.update(
            fresh_sites=list(new.sites),
            fresh_certification=new.certification,
            fresh_stage_statuses=new.stage_statuses,
            fresh_objective=a["objective"],
            reference_objective=b["objective"],
            objective_relation=rel,
            outcome=outcome,
            indicator_differences={
                k: (a["indicators"][k] - b["indicators"][k])
                if not isinstance(b["indicators"][k], str)
                else (a["indicators"][k] == b["indicators"][k])
                for k in b["indicators"]
            },
        )
        stage_ref_proven = ref.certification in (
            sol.PROVEN,
            sol.STAGE2_TIME_LIMITED,
            sol.STAGE2_NOT_PROVEN,
        )
        if rel == "better" and stage_ref_proven:
            report["inconsistencies"].append(
                f"{key}: fresh objective {a['objective']} beats the PROVEN reference {b['objective']}"
            )
        if rel == "worse" and new.certification == sol.PROVEN and ref.certification == sol.PROVEN:
            report["inconsistencies"].append(f"{key}: both runs proven optimal but objectives differ")
        if key == "individual":
            s2 = relation(a["stage2_objective"], b["stage2_objective"])
            entry.update(
                fresh_stage2_objective=a["stage2_objective"],
                reference_stage2_objective=b["stage2_objective"],
                stage2_relation=s2,
            )
            exact = alternatives.get("individual_exact_stage2")
            if exact is not None and rel == "equal":
                lower = objectives("individual", exact.sites)["stage2_objective"]
                entry["exact_stage2_optimum"] = lower
                if a["stage2_objective"] < lower - tolerance * lower:
                    report["inconsistencies"].append(
                        "individual: fresh Stage 2 objective is below the enumerated exact optimum"
                    )
                entry["stage2_note"] = (
                    "the reference Stage 2 was time-limited (not proven); a lower fresh Stage 2 value is "
                    "admissible down to the enumerated exact optimum"
                )
        report["models"][key] = entry
        log(
            f"{key}: {outcome}; certification fresh {new.certification} / reference {ref.certification}; "
            f"objective {a['objective']:.6f} vs {b['objective']:.6f}"
        )
    report["consistent"] = not report["inconsistencies"]
    (out / "compare_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "compare_report.md").write_text(_compare_markdown(report), encoding="utf-8")
    log(
        ("CONSISTENT" if report["consistent"] else "INCONSISTENT: " + "; ".join(report["inconsistencies"]))
        + f" ({out})"
    )
    return report


def compare_classified(
    workspace: Path,
    run: Path | None = None,
    other_inputs: Path | None = None,
    other_run: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Classified comparison (``comparison.compare_runs``) of this workspace's inputs (and run) with another build.

    Writes ``<workspace>/compare/classified_<n>/comparison.json`` and ``.md``.
    """
    from .comparison import compare_runs, to_markdown

    root = Path(workspace) / "compare"
    n = 1 + sum(1 for p in root.glob("classified_*")) if root.exists() else 1
    out = _new_dir(root / f"classified_{n}")
    a = {"inputs": Path(workspace) / "inputs", "run": run, "label": "this workspace"}
    b = {"inputs": other_inputs, "run": other_run, "label": "other build"}
    report = compare_runs({k: v for k, v in a.items() if v is not None}, {k: v for k, v in b.items() if v is not None})
    (out / "comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "comparison.md").write_text(to_markdown(report), encoding="utf-8")
    log(f"class {report['classification']['class']}: {report['classification']['meaning']} ({out})")
    for d in report["assumption_differences"]:
        log(f"  differs: {d['aspect']}: {d['a']} | {d['b']}")
    return report


class _NoReported:
    """Configuration stand-in without reported sites (fresh runs are never checked against them)."""

    source = Path("fresh run")

    def get(self, key: str, default: object = None) -> object:
        return {} if key == "reproducibility.reported_sites" else default


def _compare_inputs(fresh: Path, reference: Path) -> dict:
    """Do freshly built inputs equal the reference inputs (demand, access, candidates, relations, screen)?"""
    if fresh.resolve() == reference.resolve():
        return {"same_folder": True}
    a, b = (read_pickle(p / "inputs_bundle.pkl") for p in (fresh, reference))
    ra, rb = (read_pickle(p / "sparse_relations_all_demand.pkl") for p in (fresh, reference))
    da, db = a["demand"].set_index("demand_id"), b["demand"].set_index("demand_id")
    out = {
        "same_folder": False,
        "demand_locations": [len(da), len(db)],
        "same_demand_ids_and_nodes": bool(
            da.index.equals(db.index) and (da["nearest_node"] == db["nearest_node"].reindex(da.index)).all()
        ),
        "max_abs_demand_difference": float(
            (da["estimated_demand"] - db["estimated_demand"].reindex(da.index)).abs().max()
        )
        if da.index.equals(db.index)
        else None,
        "same_q4_flags": bool(
            da.index.equals(db.index) and (da["least_advantaged"] == db["least_advantaged"].reindex(da.index)).all()
        ),
    }
    ca, cb = (x["current_access"].set_index("demand_id")["current_walk_time"] for x in (a, b))
    out["max_abs_current_time_difference"] = (
        float((ca - cb.reindex(ca.index)).abs().max()) if ca.index.equals(cb.index) else None
    )
    out["same_candidates"] = list(a["candidates"]["facility_id"]) == list(b["candidates"]["facility_id"]) and list(
        a["candidates"]["nearest_node"]
    ) == list(b["candidates"]["nearest_node"])
    out["same_existing_facilities"] = sorted(a["kitas"]["nearest_node"]) == sorted(b["kitas"]["nearest_node"])
    out["relation_pairs"] = [ra.pair_count, rb.pair_count]
    out["same_relations"] = (
        ra.candidate_ids == rb.candidate_ids
        and all(
            [tuple(x) for x in ra.pairs_by_demand[i]] == [tuple(x) for x in rb.pairs_by_demand.get(i, [])]
            for i in ra.pairs_by_demand
        )
        and set(ra.pairs_by_demand) == set(rb.pairs_by_demand)
    )
    out["same_screen"] = sorted(map(int, a["screen_retained_indices"])) == sorted(
        map(int, b["screen_retained_indices"])
    )
    out["network_rule_set"] = [
        a.get("network", {}).get("version"),
        b.get("network", {}).get("version"),
    ]
    out["identical_model_inputs"] = (
        all(
            out[k]
            for k in [
                "same_demand_ids_and_nodes",
                "same_q4_flags",
                "same_candidates",
                "same_existing_facilities",
                "same_relations",
                "same_screen",
            ]
        )
        and out["max_abs_demand_difference"] is not None
        and out["max_abs_demand_difference"] < 1e-9
        and out["max_abs_current_time_difference"] is not None
        and out["max_abs_current_time_difference"] < 1e-9
    )
    return out


def _compare_markdown(report: dict) -> str:
    """Readable report of ``compare``."""
    lines = [
        "# Fresh run compared with the reference",
        "",
        f"- Run: `{report['run']}`",
        f"- Reference run: `{report['reference_run']}`",
        f"- Result: **{'consistent' if report['consistent'] else 'INCONSISTENT'}**",
        "",
        "## Inputs",
        "",
        "| Check | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in report["inputs"].items()]
    lines += [
        "",
        "## Models",
        "",
        "| Model | Outcome | Fresh certification | Reference certification | Fresh objective | Reference objective |",
        "|---|---|---|---|---:|---:|",
    ]
    for key, e in report["models"].items():
        lines.append(
            f"| {key} | {e.get('outcome')} | {e.get('fresh_certification', '')} | {e.get('reference_certification')} | "
            f"{e.get('fresh_objective', '')} | {e.get('reference_objective', '')} |"
        )
    ind = report["models"].get("individual", {})
    if "stage2_relation" in ind:
        lines += [
            "",
            f"Individual Rawlsian Stage 2: fresh {ind['fresh_stage2_objective']:.4f}, reference "
            f"{ind['reference_stage2_objective']:.4f} "
            f"({ind['stage2_relation']}); exact optimum {ind.get('exact_stage2_optimum', 'n/a')}. "
            f"{ind.get('stage2_note', '')}",
        ]
    if report["inconsistencies"]:
        lines += ["", "## Inconsistencies", ""] + [f"- {x}" for x in report["inconsistencies"]]
    return "\n".join(lines) + "\n"
