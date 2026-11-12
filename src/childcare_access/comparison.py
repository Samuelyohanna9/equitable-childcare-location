"""Fingerprints of built model inputs, solver-status tables and comparisons of runs.

``fingerprint_inputs`` hashes the canonical content of every solver input in an inputs folder (exact SHA-256: same
hash = same values, bit for bit) and keeps numerical summaries (row counts, column sums) for reports. Exact
equality and numerical equivalence within a tolerance are always reported separately; a numerically close input
is never called identical.

``compare_runs`` classifies two builds (with or without solved runs) by their CONTENT, never by names or paths:

- **A**, identical standardised solver inputs: results are compared directly (objectives, certification, access);
- **B**, same city (same district geometry), different inputs: a sensitivity analysis; every differing assumption
  is listed first, and each run is evaluated on its own inputs;
- **C**, different cities: descriptive summaries only; objectives, access levels and index values are not compared.

``status_table`` gives the literal solver records of a run with their certification (proven optimal, within the
requested gap, time-limited incumbent, no solution) and any independent exact reference, kept apart.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import shapely

if TYPE_CHECKING:  # imported lazily at run time: these modules read the active configuration
    import networkx as nx

    from .all_demand_candidates import SparseCandidateRelations

COMPONENTS = {
    "districts": [
        "district_id",
        "district_name",
        "childcare_vulnerability_index",
        "vulnerability_quartile",
        "least_advantaged",
        "geometry",
    ],
    "district_groups": [
        "district_name",
        "childcare_vulnerability_index",
        "vulnerability_quartile",
        "least_advantaged_threshold",
        "least_advantaged_quantile",
        "least_advantaged",
    ],
    "children": ["district_id", "age_0_2", "age_3_5", "children_0_5"],
    "demand": [
        "demand_id",
        "nearest_node",
        "district_id",
        "district_name",
        "x",
        "y",
        "estimated_demand",
        "age_0_2",
        "age_3_5",
        "childcare_vulnerability_index",
        "vulnerability_quartile",
        "least_advantaged",
    ],
    "candidates": ["facility_id", "nearest_node", "base_node", "demand_id", "district_name"],
    "facilities": ["facility_id", "nearest_node", "base_node"],
    "current_access": ["demand_id", "nearest_existing_kita", "current_walk_time"],
    "district_geometry": [
        "district_id",
        "district_name",
        "geometry",
    ],  # the city itself, without any group
}
BUNDLE_KEY = {"facilities": "kitas", "district_geometry": "districts"}
SORT = {
    "districts": "district_id",
    "district_groups": "district_name",
    "children": "district_id",
    "demand": "demand_id",
    "candidates": "facility_id",
    "facilities": "facility_id",
    "current_access": "demand_id",
    "district_geometry": "district_id",
}
# Quantities summed in the numerical summary (identifiers and node numbers are not summed).
SUMMED = {
    "districts": ["childcare_vulnerability_index", "least_advantaged"],
    "children": ["age_0_2", "age_3_5", "children_0_5"],
    "demand": ["estimated_demand", "age_0_2", "age_3_5", "least_advantaged"],
    "current_access": ["current_walk_time"],
    "district_groups": ["least_advantaged"],
    "candidates": [],
    "facilities": [],
    "district_geometry": [],
}
FILES = [
    "inputs_bundle.pkl",
    "sparse_relations_all_demand.pkl",
    "walk_graph_base.pkl",
    "walk_graph_layered.pkl",
]
CLASSES = {
    "A": "identical standardised solver inputs: results are compared directly",
    "B": "same city (same district geometry), different standardised inputs: a sensitivity analysis - the results "
    "differ because the assumptions listed below differ",
    "C": "different cities: descriptive summaries only; objectives, access levels and vulnerability values are not "
    "directly comparable across cities",
}


def _sha(text: str) -> str:
    """SHA-256 of a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _table(frame: pd.DataFrame, name: str) -> str:
    """CSV with shortest round-trip float formatting: equal text <=> equal values."""
    columns = [c for c in COMPONENTS[name] if c in frame.columns]  # absent optional columns (no age split)
    frame = pd.DataFrame(frame)[columns].copy()
    if "geometry" in columns:
        frame["geometry"] = shapely.to_wkb(frame["geometry"].to_numpy(), hex=True)
    return frame.sort_values(SORT[name], kind="stable").to_csv(index=False, lineterminator="\n")


def _graph(graph: nx.Graph) -> str:
    """Canonical text of a graph (nodes, edges and their plain attributes), for hashing."""

    def plain(data: dict) -> list:
        """The bool, number and text attributes of a node or edge, sorted."""
        return sorted((k, v) for k, v in data.items() if isinstance(v, (bool, int, float, str)))

    nodes = sorted((n, plain(d)) for n, d in graph.nodes(data=True))
    edges = sorted(
        (repr(e[:-1]), plain(e[-1]))
        for e in (graph.edges(keys=True, data=True) if graph.is_multigraph() else graph.edges(data=True))
    )
    return repr(nodes) + "\n" + repr(edges)


def _relations(relations: SparseCandidateRelations) -> str:
    """Canonical text of the sparse relations (candidates and the improving pairs of every location)."""
    pairs = relations.pairs_by_demand
    return repr((relations.candidate_ids, relations.candidate_nodes, [(d, pairs[d]) for d in sorted(pairs)]))


def _bundle(folder: Path) -> dict:
    """The input bundle of an input folder."""
    with (Path(folder) / "inputs_bundle.pkl").open("rb") as handle:
        return pickle.load(handle)


def fingerprint_inputs(folder: str | Path) -> dict:
    """Exact content hashes and numerical summaries of the solver inputs in ``folder`` (read only)."""
    folder = Path(folder)
    bundle = _bundle(folder)
    with (folder / "sparse_relations_all_demand.pkl").open("rb") as handle:
        relations = pickle.load(handle)
    tables = {name: bundle[BUNDLE_KEY.get(name, name)] for name in COMPONENTS}
    exact = {name: _sha(_table(frame, name)) for name, frame in tables.items()}
    exact["relations"] = _sha(_relations(relations))
    exact["screen"] = _sha(repr(sorted(map(int, bundle["screen_retained_indices"]))))
    for name in ("walk_graph_base.pkl", "walk_graph_layered.pkl"):
        with (folder / name).open("rb") as handle:
            exact[name.removesuffix(".pkl")] = _sha(_graph(pickle.load(handle)))
    exact["labels"] = _sha(
        repr(
            [
                bundle["scenario"],
                bundle["allocation_variant"],
                bundle["building_demand_column"],
                sorted(bundle["demand"]["demand_definition"].unique()),
                bundle["network"]["version"],
            ]
        )
    )

    summary = {
        name: {
            "rows": len(frame),
            **{c: float(pd.to_numeric(frame[c]).sum()) for c in SUMMED[name] if c in frame},
        }
        for name, frame in tables.items()
    }
    summary["relations"] = {
        "pairs": relations.pair_count,
        "candidates": len(relations.candidate_ids),
        "time_sum": float(sum(t for p in relations.pairs_by_demand.values() for _, t in p)),
    }
    summary["screen"] = {"retained": len(bundle["screen_retained_indices"])}
    files = {}
    for name in FILES:
        digest = hashlib.sha256()
        with (folder / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        files[name] = digest.hexdigest()
    return {"exact": exact, "summary": summary, "file_sha256": files}


# classification
def classify(a: dict, b: dict) -> dict:
    """Comparison class of two fingerprints (``fingerprint_inputs``): by content, never by names or paths."""
    scientific = [k for k in a["exact"] if k not in ("labels", "district_geometry") and k in b["exact"]]
    differing = [k for k in scientific if a["exact"][k] != b["exact"][k]]
    same_city = a["exact"]["district_geometry"] == b["exact"]["district_geometry"]
    kind = "A" if not differing and same_city else ("B" if same_city else "C")
    return {
        "class": kind,
        "meaning": CLASSES[kind],
        "differing_components": differing,
        "same_district_geometry": same_city,
        "labels_identical": a["exact"]["labels"] == b["exact"]["labels"],
        "file_bytes_identical": {k: a["file_sha256"][k] == b["file_sha256"][k] for k in a["file_sha256"]},
    }


def assumptions(inputs: str | Path, run: str | Path | None = None) -> dict:
    """The assumptions behind a build (and run).

    Taken from its provenance (version 2), or derived from the inputs bundle for version-1 builds and the saved
    Münster model inputs (whose provenance is incomplete).
    """
    from . import provenance as prov
    from . import vulnerability_modes as vm

    bundle, record = _bundle(Path(inputs)), prov.load(inputs)
    groups = bundle["district_groups"]
    priority = bundle.get("priority_group") or {
        "mode": "four_indicator",
        "quantile": float(groups["least_advantaged_quantile"].iloc[0]),
        "selected": sorted(groups.loc[groups["least_advantaged"].astype(bool), "district_name"]),
    }
    out = {
        "provenance": "version-2 provenance record"
        if record and "incomplete" not in record
        else (
            "INCOMPLETE: " + ", ".join(record["incomplete"])
            if record
            else "INCOMPLETE: no provenance record (version-1 build); assumptions derived from the inputs bundle"
        ),
        "P": None,
        "demand definition": sorted(bundle["demand"]["demand_definition"].unique()),
        "children": float(bundle["demand"]["estimated_demand"].sum()),
        "priority group": priority.get("label") or vm.label(priority),
        "priority districts": priority["selected"],
        "candidates": {
            "source": sorted(bundle["candidates"]["candidate_source"].astype(str).unique()),
            "count": len(bundle["candidates"]),
        },
        "network rule set": bundle["network"]["version"],
    }
    if record:
        demand = record["demand"]
        out["population"] = {
            k: demand.get(k) for k in ("mode", "count_type", "reference_year", "census_year", "source")
        }
        out["walking speed (m/min)"] = record["parameters"].get("walking_speed_m_per_min")
        out["network snapshot"] = record["network"].get("snapshot_timestamp")
        out["crs"] = record["crs"]
    if run is not None:
        manifest = sorted(Path(run).glob("run_manifest*.json"))
        if manifest:
            out["P"] = json.loads(manifest[-1].read_text(encoding="utf-8")).get("facility_budget")
    return out


def assumption_differences(a: dict, b: dict) -> list[dict]:
    """Assumptions recorded for both builds that differ.

    An assumption recorded for only one build (missing provenance) is unknown, not different: see
    ``unrecorded_assumptions``.
    """
    keys = [k for k in a if k != "provenance" and a.get(k) is not None and b.get(k) is not None]
    return [{"aspect": k, "a": a[k], "b": b[k]} for k in keys if a[k] != b[k]]


def unrecorded_assumptions(a: dict, b: dict) -> list[str]:
    """Assumptions recorded for one build but not for the other."""
    return sorted(k for k in set(a) | set(b) if k != "provenance" and (a.get(k) is None) != (b.get(k) is None))


# solver status
def _category(record: dict[str, Any]) -> str:
    """Certification category of one solver record (proven, within gap, no solution)."""
    from .solutions import NO_SOLUTION, PROVEN, WITHIN_GAP, certify_stage

    stage = certify_stage(record)
    if stage == NO_SOLUTION:
        return f"no solution ({record.get('status')}: infeasible or failed)"
    if stage == PROVEN:
        return "proven optimal"
    if record.get("status") == "TimeLimit":
        return "time-limited: feasible incumbent, NOT proven optimal"
    if stage == WITHIN_GAP:
        return "feasible incumbent within the requested MIP gap, NOT proven optimal"
    return f"feasible incumbent ({record.get('status')}), NOT proven optimal"


def status_table(run: str | Path) -> pd.DataFrame:
    """Literal solver records of a run, one row per solver call, with the certification category.

    An independent exact reference (if saved) is a separate row, marked as not reported. Nothing is filled in that
    the records lack.
    """
    from . import solutions as sol

    rows = []
    for path in sorted(Path(run).glob("*/*_solver_records.csv")):
        for record in pd.read_csv(path).to_dict("records"):
            rows.append(
                {
                    "model": path.parent.name,
                    "role": record["role"],
                    "solver_status": record.get("status"),
                    "certification": _category(record),
                    "objective": record.get("incumbent_objective"),
                    "best_bound": record.get("best_bound"),
                    "mip_gap": record.get("mip_gap"),
                    "runtime_s": record.get("runtime_seconds"),
                    "time_limit_s": record.get("time_limit_seconds"),
                    "reported": True,
                }
            )
    for key, solution in sol.load_alternative_solutions(run_dir=run, config=_NoSites()).items():
        details = Path(run) / "rawlsian_exact_stage2" / "rawlsian_result.json"
        details = json.loads(details.read_text(encoding="utf-8")) if details.is_file() else {}
        rows.append(
            {
                "model": key,
                "role": solution.label + " - Stage 2 objective",
                "solver_status": solution.status,
                "certification": "independent exact reference ("
                + solution.certification
                + "), NOT the reported result",
                "objective": details.get("stage2_exact_optimum"),
                "best_bound": None,
                "mip_gap": None,
                "runtime_s": None,
                "time_limit_s": None,
                "reported": False,
            }
        )
    return pd.DataFrame(rows)


class _NoSites:
    """Configuration stand-in: runs are compared as saved, never checked against reported sites."""

    source = Path("comparison")

    def get(self, key: str, default: object = None) -> object:
        return {} if key == "reproducibility.reported_sites" else default


# run comparison
def _evaluate(inputs: Path, solutions: dict, group: str) -> dict:
    """Objectives as saved and accessibility indicators recomputed on ``inputs`` for each model's sites."""
    from .evaluation import accessibility_indicators, scenario_times

    bundle = _bundle(inputs)
    with (inputs / bundle["network"]["routing_graph"]).open("rb") as handle:
        graph = pickle.load(handle)
    demand = bundle["demand"].set_index("demand_id")
    nodes = bundle["candidates"].set_index("facility_id")["nearest_node"]
    out = {}
    for key, s in solutions.items():
        times = scenario_times(graph, demand, bundle["kitas"]["nearest_node"], nodes, s.sites)
        out[key] = {
            "sites": list(s.sites),
            "certification": s.certification,
            "objective": s.objective_value,
            "indicators": accessibility_indicators(times, demand, group=group),
        }
    return out


def compare_runs(a: dict, b: dict) -> dict:
    """Compare two builds ``{"inputs": folder, "run": folder or None, "label": text}``.

    Indicator keys keep their meaning: "... mean" is unweighted over demand locations, "... demand-weighted mean"
    weights by estimated children.
    """
    from . import solutions as sol
    from . import vulnerability_modes as vm

    a_in, b_in = Path(a["inputs"]), Path(b["inputs"])
    fa, fb = fingerprint_inputs(a_in), fingerprint_inputs(b_in)
    found = classify(fa, fb)
    sa, sb = assumptions(a_in, a.get("run")), assumptions(b_in, b.get("run"))
    differences = assumption_differences(sa, sb)
    if sa["P"] is not None and sb["P"] is not None and sa["P"] != sb["P"] and found["class"] == "A":
        found = {
            **found,
            "class": "B",
            "meaning": CLASSES["B"] + " (identical inputs, different P)",
        }
    report = {
        "a": {k: str(v) for k, v in a.items()},
        "b": {k: str(v) for k, v in b.items()},
        "classification": found,
        "assumptions": {"a": sa, "b": sb},
        "assumption_differences": differences,
        "unrecorded_in_one_build": unrecorded_assumptions(sa, sb),
        "metric_note": "'mean' = unweighted over demand locations; 'demand-weighted mean' = weighted by estimated "
        "children; the priority group is a district classification",
    }
    if not (a.get("run") and b.get("run")):
        return report
    runs = {}
    for side, spec, assume in (("a", a, sa), ("b", b, sb)):
        solutions = sol.load_reported_solutions(run_dir=spec["run"], config=_NoSites(), require_all=False)
        group = "Q4" if assume["priority group"].startswith("Q4") else vm.group_name({"mode": "priority_districts"})
        runs[side] = {
            "status": status_table(spec["run"]).to_dict("records"),
            "results": _evaluate(Path(spec["inputs"]), solutions, group)
            if found["class"] != "C"
            else {
                k: {
                    "certification": s.certification,
                    "objective (this city only)": s.objective_value,
                }
                for k, s in solutions.items()
            },
        }
    report["runs"] = runs
    if found["class"] == "A":  # same inputs: direct differences
        report["differences"] = {
            key: {
                "objective_difference": runs["a"]["results"][key]["objective"] - runs["b"]["results"][key]["objective"],
                "same_sites": sorted(runs["a"]["results"][key]["sites"]) == sorted(runs["b"]["results"][key]["sites"]),
                "certification": [
                    runs["a"]["results"][key]["certification"],
                    runs["b"]["results"][key]["certification"],
                ],
                "indicator_differences": {
                    k: v - runs["b"]["results"][key]["indicators"][k]
                    for k, v in runs["a"]["results"][key]["indicators"].items()
                    if not isinstance(v, str)
                },
            }
            for key in runs["a"]["results"]
            if key in runs["b"]["results"]
        }
    return report


def to_markdown(report: dict) -> str:
    """Readable report: class and assumption differences first (for the notebook and the CLI)."""
    c = report["classification"]
    lines = [
        f"# Comparison: class {c['class']}",
        "",
        c["meaning"],
        "",
        f"- inputs A: `{report['a'].get('inputs')}`; inputs B: `{report['b'].get('inputs')}`",
        f"- differing solver-input components: {', '.join(c['differing_components']) or 'none'}",
        f"- descriptive labels identical: {c['labels_identical']}; file bytes identical: "
        f"{', '.join(k for k, v in c['file_bytes_identical'].items() if v) or 'none'} (pickles are not "
        "byte-reproducible)",
        f"- provenance A: {report['assumptions']['a']['provenance']}",
        f"- provenance B: {report['assumptions']['b']['provenance']}",
        "",
        "## Differing assumptions",
        "",
    ]
    lines += [f"- **{d['aspect']}**: A = {d['a']} | B = {d['b']}" for d in report["assumption_differences"]] or [
        "- none"
    ]
    if report.get("unrecorded_in_one_build"):
        lines += [
            f"- recorded for one build only (unknown for the other): {', '.join(report['unrecorded_in_one_build'])}"
        ]
    for side in ("a", "b"):
        if "runs" not in report:
            break
        lines += [
            "",
            f"## Solver status, run {side.upper()}",
            "",
            "| model | role | status | certification | objective | gap |",
            "|---|---|---|---|---:|---:|",
        ]
        lines += [
            f"| {r['model']} | {r['role']} | {r['solver_status']} | {r['certification']} | {r['objective']} | "
            f"{r['mip_gap']} |"
            for r in report["runs"][side]["status"]
        ]
    if "differences" in report:
        lines += ["", "## Direct differences (identical inputs)", ""]
        lines += [
            f"- {k}: objective A - B = {v['objective_difference']:.6g}; same sites: {v['same_sites']}; "
            f"certification {v['certification'][0]} / {v['certification'][1]}"
            for k, v in report["differences"].items()
        ]
    lines += ["", report["metric_note"]]
    return "\n".join(lines) + "\n"
