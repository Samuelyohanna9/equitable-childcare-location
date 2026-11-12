"""Read a built inputs folder, and the result folders named by the city configuration (no solving).

``load_primary_inputs`` reads any inputs folder written by ``childcare-access build`` (and the saved Münster model
inputs). ``INPUTS_FINAL``, ``RUN_FINAL`` and ``AUDIT_FINAL`` are the ``[outputs]`` folders of the active
configuration; the Münster example notebook reads them.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import geopandas as gpd

from .core import PROJECTED_CRS
from .settings import CONFIG

# Reference inputs and results of the active city configuration ([outputs]).
INPUTS_FINAL = CONFIG.path("outputs.final_inputs")
RUN_FINAL = CONFIG.path("outputs.final_run")
AUDIT_FINAL = CONFIG.path("outputs.audit")


def load_primary_inputs(inputs_dir: str | Path = INPUTS_FINAL, network: bool = True) -> dict:
    """Validated inputs: districts, Q4 groups, buildings, Kitas, network, demand, access, candidates, relations."""
    inputs_dir = Path(inputs_dir)
    report = json.loads((inputs_dir / "validation_report.json").read_text(encoding="utf-8"))
    if not report["all_checks_passed"]:
        raise RuntimeError(f"Input validation in {inputs_dir} did not pass.")
    with (inputs_dir / "inputs_bundle.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    with (inputs_dir / "sparse_relations_all_demand.pkl").open("rb") as handle:
        relations = pickle.load(handle)

    buildings = gpd.read_file(inputs_dir / "buildings_alkis_allocated.gpkg", ignore_geometry=True)
    # The bundle names the building-level demand column.
    buildings["estimated_demand"] = buildings[bundle.get("building_demand_column") or "children_0_5_footprint"]
    buildings = gpd.GeoDataFrame(
        buildings,
        geometry=gpd.points_from_xy(buildings["rep_x"], buildings["rep_y"]),
        crs=PROJECTED_CRS,
    )
    graph = graph_base = None
    if network:
        with (inputs_dir / bundle["network"]["routing_graph"]).open("rb") as handle:
            graph = pickle.load(handle)
        with (inputs_dir / bundle["network"]["base_graph"]).open("rb") as handle:
            graph_base = pickle.load(handle)
    return {
        "graph_base": graph_base,
        "districts": bundle["districts"],
        "district_vulnerability": bundle["district_groups"],
        "children": bundle["children"],
        "residential_buildings": buildings,
        "existing_kitas": bundle["kitas"],
        "graph": graph,
        "demand": bundle["demand"],
        "current_access": bundle["current_access"],
        "candidate_sites": bundle["candidates"],
        "relations": relations,
        "screen_retained_indices": bundle["screen_retained_indices"],
        "validation_report": report,
    }
