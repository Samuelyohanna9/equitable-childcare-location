"""Full workflow on the synthetic example city: build -> solve -> compare, through the CLI.

The city is examples/synthetic_city/make_city.py with config/synthetic_city.toml, run in a clean temporary
workspace. No Münster data or results are used.

The build tests need no Gurobi. The solve tests (marker ``gurobi``) check every model against EXHAUSTIVE ENUMERATION
of all site sets, for P = 1, 2 and 3, with walking times recomputed independently by Dijkstra:
- Utilitarian: minimum of sum_i D_i t_i (demand weighting; capacity-free nearest-open assignment);
- Individual Rawlsian Stage 1: minimum of the worst least-advantaged walking time;
- Individual Rawlsian Stage 2: minimum squared deviation among the Stage 1 optimal site sets (lexicographic);
- District Rawlsian: minimum of the worst least-advantaged district's demand-weighted mean, over ALL candidates
  (so the dominance screen is validated too).
Time-limited runs, infeasible budgets, missing data and protected folders are tested as well.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from childcare_access import evaluation
from childcare_access import solutions as sol
from childcare_access.config import CONFIG_ENV, PROJECT_ROOT, ConfigError

sys.path.insert(0, str(PROJECT_ROOT / "examples" / "synthetic_city"))
from make_city import CHILDREN, DISTRICTS, EXPECTED_Q4, INDICATORS, X0, Y0, write_data

from childcare_access.script_support import read_pickle


def write_city(root: Path) -> Path:
    """The committed example configuration, pointed at freshly generated data and a temporary workspace."""
    data = write_data(root / "data")
    text = (PROJECT_ROOT / "config" / "synthetic_city.toml").read_text(encoding="utf-8")
    text = text.replace('"examples/synthetic_city/data/', f'"{data.as_posix()}/')
    text = text.replace('"examples/synthetic_city/reference/', f'"{(root / "reference").as_posix()}/')
    text = text.replace('root = "workspace/synthetic_city"', f'root = "{(root / "workspace").as_posix()}"')
    text = text.replace('nodefile_dir = ""', f'nodefile_dir = "{(root / "nodefiles").as_posix()}"')
    assert not any("examples/synthetic_city" in line for line in text.splitlines() if not line.lstrip().startswith("#"))
    path = root / "synthetic_city.toml"
    path.write_text(text, encoding="utf-8")
    return path


TOL = 1e-6


def cli(config: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, CONFIG_ENV: str(config), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, "-m", "childcare_access", *args, "--config", str(config)],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture(scope="module")
def city(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("synthetic_city")
    config = write_city(root)
    workspace = root / "workspace"
    run = cli(config, "build", "--workspace", str(workspace))
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    inputs = workspace / "inputs"
    bundle = read_pickle(inputs / "inputs_bundle.pkl")
    graph = read_pickle(inputs / "walk_graph_layered.pkl")
    return {
        "root": root,
        "config": config,
        "workspace": workspace,
        "inputs": inputs,
        "bundle": bundle,
        "graph": graph,
        "build_log": run.stdout,
    }


# ================================================================================================ build (no Gurobi)
def test_build_report_and_provenance(city):
    report = json.loads((city["inputs"] / "validation_report.json").read_text(encoding="utf-8"))
    assert (
        report["all_checks_passed"]
        and report["built_by"] == "childcare-access build"
        and report["config"] == "synthetic_city.toml"
    )
    assert {
        "raw input files present (recorded checksums match)",
        "demand locations conserve every district total",
        "optimisation inputs feasible",
    } <= {c["check"] for c in report["checks"]}


def test_no_muenster_data_or_results_are_used(city):
    """Every raw input and the facility source lie in the synthetic city's own folder; no reference result is read."""
    report = json.loads((city["inputs"] / "validation_report.json").read_text(encoding="utf-8"))
    data = (city["root"] / "data").as_posix()
    assert all(Path(v["path"]).as_posix().startswith(data) for v in report["raw_inputs"].values()), report["raw_inputs"]
    assert Path(report["kita_source"]).as_posix().startswith(data)
    assert report["indicator_definitions"] and not (city["root"] / "reference").exists()


def test_q4_classification_is_recomputed_independently(city):
    """Equal-weight min-max index of the four indicators; Q4 = index at or above the configured 50th percentile."""
    values = pd.DataFrame(INDICATORS).T.astype(float)
    index = ((values - values.min()) / (values.max() - values.min())).mean(axis=1)
    expected = sorted(DISTRICTS[k][0] for k in index.index[index >= index.quantile(0.5)])
    groups = city["bundle"]["district_groups"]
    assert (
        expected == EXPECTED_Q4 == sorted(groups.loc[groups["least_advantaged_district"].astype(bool), "district_name"])
    )
    demand = city["bundle"]["demand"]
    assert (demand["least_advantaged"].astype(bool) == demand["district_name"].isin(EXPECTED_Q4)).all()


def test_demand_conserves_the_forecast_of_every_district(city):
    demand = city["bundle"]["demand"]
    allocated = demand.groupby("district_name")["estimated_demand"].sum()
    for nr, (young, older) in CHILDREN.items():
        assert allocated[DISTRICTS[nr][0]] == pytest.approx(young + older, abs=1e-9)
    assert (
        any(str(b).startswith("census_cell:") for b in demand["represented_buildings"])
        or len(pd.read_csv(city["inputs"] / "fallback_census_cells.csv")) == 1
    )  # one fallback origin (cell without a building)


def test_network_travel_times_follow_the_grid(city):
    """On the 100 m footway grid the walk between two grid nodes is their Manhattan distance / 83.33 m per minute.

    (OSMnx simplification removes the four degree-2 corners, so the Kita attaches to a node next to its corner.)
    """
    from pyproj import Transformer

    graph, bundle = city["graph"], city["bundle"]
    to_utm = Transformer.from_crs(4326, 25832, always_xy=True)

    def grid(node):
        x, y = to_utm.transform(graph.nodes[int(node)]["x"], graph.nodes[int(node)]["y"])
        return round((x - X0) / 100), round((y - Y0) / 100)

    kita = grid(bundle["kitas"]["nearest_node"].iloc[0])
    current = bundle["current_access"].set_index("demand_id")["current_walk_time"]
    for _, row in bundle["demand"].iterrows():
        i, j = grid(row["nearest_node"])
        manhattan_m = 100 * (abs(i - kita[0]) + abs(j - kita[1]))
        # OSMnx measures great-circle edge lengths; the grid is 100 m in UTM (scale factor), hence 0.5% tolerance
        assert current[row["demand_id"]] == pytest.approx(manhattan_m / 83.33, rel=5e-3), row["demand_id"]


def test_the_workspace_never_overlaps_protected_folders():
    from childcare_access import workflow

    for folder in ["data", "results", "results/muenster/new", "config", "src", "docs", "examples"]:
        with pytest.raises(ConfigError, match="overlaps the protected folder"):
            workflow.workspace_root(PROJECT_ROOT / folder)
    with pytest.raises(ConfigError, match="overlaps the protected folder"):
        workflow.workspace_root(PROJECT_ROOT)  # the project root contains the protected folders


def test_build_never_overwrites_and_reports_missing_data(city, tmp_path):
    again = cli(city["config"], "build", "--workspace", str(city["workspace"]))
    assert again.returncode != 0 and "already contains results" in again.stderr
    text = city["config"].read_text(encoding="utf-8").replace("forecast.csv", "absent.csv")
    broken = tmp_path / "broken.toml"
    broken.write_text(text, encoding="utf-8")
    run = cli(broken, "build", "--workspace", str(tmp_path / "ws"))
    assert run.returncode != 0 and "absent.csv" in run.stderr


def test_an_infeasible_budget_is_refused_before_solving(city):
    run = cli(
        city["config"],
        "solve",
        "--workspace",
        str(city["workspace"]),
        "--p",
        "99",
        "--run-name",
        "infeasible",
    )
    assert run.returncode != 0 and "P = 99 exceeds the 16 candidate sites" in run.stderr
    assert not (city["workspace"] / "runs" / "infeasible").exists()


# ===================================================================================== solve (Gurobi) vs enumeration
def _enumerate(city, P: int) -> pd.DataFrame:
    bundle, graph = city["bundle"], city["graph"]
    demand = bundle["demand"].set_index("demand_id")
    nodes = bundle["candidates"].set_index("facility_id")["nearest_node"]
    kitas = bundle["kitas"]["nearest_node"]
    q4, D = demand["least_advantaged"].astype(bool), demand["estimated_demand"]
    rows = []
    for sites in combinations(sorted(nodes.index), P):
        t = evaluation.scenario_times(graph, demand, kitas, nodes, sites)
        rows.append(
            {
                "sites": sites,
                "utilitarian": float((t * D).sum()),
                "z": float(t[q4].max()),
                "stage2": evaluation.lexicographic_stage2_objective(t, demand),
                "district": float(evaluation.district_means(t, demand).max()),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def solved(city) -> dict:
    out = {}
    for P in (1, 2, 3):
        run = cli(
            city["config"],
            "solve",
            "--workspace",
            str(city["workspace"]),
            "--p",
            str(P),
            "--run-name",
            f"P{P}",
        )
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        out[P] = {
            "dir": city["workspace"] / "runs" / f"P{P}",
            "enumeration": _enumerate(city, P),
            "solutions": sol.load_reported_solutions(
                run_dir=city["workspace"] / "runs" / f"P{P}", config=_NoSites(), require_all=False
            ),
        }
    return out


class _NoSites:
    source = Path("synthetic")

    def get(self, key, default=None):
        return {} if key == "reproducibility.reported_sites" else default


@pytest.mark.gurobi
@pytest.mark.parametrize("P", [1, 2, 3])
def test_utilitarian_matches_enumeration(solved, P):
    s, e = solved[P]["solutions"]["utilitarian"], solved[P]["enumeration"]
    assert s.certification == sol.PROVEN
    assert s.objective_value == pytest.approx(e["utilitarian"].min(), rel=TOL)
    assert e.set_index("sites").loc[[tuple(sorted(s.sites))], "utilitarian"].iloc[0] == pytest.approx(
        e["utilitarian"].min(), rel=TOL
    )


@pytest.mark.gurobi
@pytest.mark.parametrize("P", [1, 2, 3])
def test_individual_rawlsian_both_stages_match_enumeration(solved, P):
    s, e = solved[P]["solutions"]["individual"], solved[P]["enumeration"]
    z_star = e["z"].min()
    assert s.stage_statuses["Individual Rawlsian Stage 1"] == "Optimal" and s.objective_value == pytest.approx(
        z_star, rel=TOL
    )
    stage1_optimal = e[e["z"] <= z_star + TOL]
    chosen = e.set_index("sites").loc[[tuple(sorted(s.sites))]].iloc[0]
    assert chosen["z"] == pytest.approx(z_star, rel=TOL)  # the Stage 2 sites keep Z*
    assert s.certification == sol.PROVEN  # tiny instance: Stage 2 proven too
    assert chosen["stage2"] == pytest.approx(stage1_optimal["stage2"].min(), rel=TOL)  # lexicographic optimum


@pytest.mark.gurobi
@pytest.mark.parametrize("P", [1, 2, 3])
def test_district_rawlsian_matches_enumeration_over_all_candidates(solved, P):
    s, e = solved[P]["solutions"]["district"], solved[P]["enumeration"]
    assert s.certification == sol.PROVEN
    assert s.objective_value == pytest.approx(
        e["district"].min(), rel=TOL
    )  # screened pool reaches the full-pool optimum


@pytest.mark.gurobi
def test_assignment_is_capacity_free_nearest_open_and_runs_are_recorded(city, solved):
    for P, run in solved.items():
        manifest = json.loads((run["dir"] / "run_manifest_solve.json").read_text(encoding="utf-8"))
        assert (
            manifest["facility_budget"] == P
            and manifest["reference_solutions_read"] is False
            and manifest["gurobi_version"]
        )
        for model in ("utilitarian", "rawlsian", "district"):
            validation = pd.read_csv(run["dir"] / model / f"{model}_independent_validation.csv")
            column = "mismatch_count" if "mismatch_count" in validation else "modeled_vs_independent_mismatch_count"
            assert int(validation[column].iloc[0]) == 0  # every location at its nearest open facility
            records = pd.read_csv(run["dir"] / model / f"{model}_solver_records.csv")
            assert {
                "seed",
                "runtime_seconds",
                "best_bound",
                "mip_gap",
                "time_limit_seconds",
                "certification",
            } <= set(records.columns)


# =========================================================================================== time-limited and compare
def _solve_in_process(city, run_name: str, overrides: dict, models=("rawlsian",)) -> Path:
    code = (
        "import sys, pathlib; from childcare_access import workflow\n"
        f"workflow.solve(pathlib.Path(r'{city['workspace']}'), run_name={run_name!r}, models={models!r}, "
        f"solver_overrides={overrides!r}, log=lambda m: None)"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, CONFIG_ENV: str(city["config"])},
    )
    assert run.returncode == 0, run.stderr[-3000:]
    return city["workspace"] / "runs" / run_name


@pytest.mark.gurobi
def test_a_time_limited_stage2_is_never_labelled_proven(city, solved):
    """A time-limited Stage 2 is never labelled proven.

    Stage 2 reaches its time limit as soon as the Stage 1 start is accepted (deterministic test hook). Stage 1 stays
    proven; the Stage 2 incumbent is not the Stage 2 optimum.
    """
    out = _solve_in_process(
        city,
        "stage2_time_limited",
        {"all_demand_candidates_rawlsian_stage2": {"stop_at_first_solution": True}},
    )
    s = sol.load_reported_solutions(run_dir=out, config=_NoSites(), require_all=False)["individual"]
    statuses = list(s.stage_statuses.values())
    assert statuses == ["Optimal", "TimeLimit"]
    assert s.certification == sol.STAGE2_TIME_LIMITED and not s.proven_optimal
    assert sorted(s.sites) == sorted(s.provenance["stage1_sites"])  # the incumbent is the Stage 1 start
    e = solved[3]["enumeration"]
    assert s.objective_value == pytest.approx(e["z"].min(), rel=TOL)  # Stage 1 still proven
    stage2 = e.set_index("sites").loc[[tuple(sorted(s.sites))], "stage2"].iloc[0]
    assert stage2 > e[e["z"] <= e["z"].min() + TOL]["stage2"].min() + TOL  # here the incumbent is NOT Stage 2 optimal


@pytest.mark.gurobi
def test_a_time_limited_district_run_is_feasible_not_proven(city):
    out = _solve_in_process(
        city,
        "district_time_limited",
        {"q4_district_estimated_demand_maximin": {"stop_at_first_solution": True}},
        models=("utilitarian", "district"),
    )
    s = sol.load_reported_solutions(run_dir=out, config=_NoSites(), require_all=False)["district"]
    assert s.stage_statuses["District Rawlsian (screened)"] == "TimeLimit"
    assert s.certification == sol.FEASIBLE and not s.proven_optimal


@pytest.mark.gurobi
def test_compare_labels_identical_and_not_comparable_runs(city, solved):
    workspace, reference = city["workspace"], str(city["workspace"] / "runs" / "P3")
    again = cli(
        city["config"],
        "solve",
        "--workspace",
        str(workspace),
        "--p",
        "3",
        "--run-name",
        "P3_again",
        "--district-warm-start",
        "none",
    )
    assert again.returncode == 0, again.stderr[-2000:]
    same = cli(
        city["config"],
        "compare",
        "--workspace",
        str(workspace),
        "--run-name",
        "P3_again",
        "--reference-run",
        reference,
        "--reference-inputs",
        str(city["inputs"]),
    )
    assert same.returncode == 0 and "CONSISTENT" in same.stdout
    report = json.loads((workspace / "compare" / "P3_again" / "compare_report.json").read_text(encoding="utf-8"))
    for model in ("utilitarian", "individual", "district"):
        entry = report["models"][model]
        assert entry["objective_relation"] == "equal"
        assert entry["outcome"] in (
            "identical sites",
            "alternative optimum (same objective, other sites)",
        )
    other = cli(
        city["config"],
        "compare",
        "--workspace",
        str(workspace),
        "--run-name",
        "P2",
        "--reference-run",
        reference,
        "--reference-inputs",
        str(city["inputs"]),
    )
    report = json.loads((workspace / "compare" / "P2" / "compare_report.json").read_text(encoding="utf-8"))
    assert other.returncode == 0 and all(e["outcome"].startswith("not comparable") for e in report["models"].values())
