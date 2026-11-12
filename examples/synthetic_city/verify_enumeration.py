"""Independent check of the synthetic city's solved runs: exhaustive enumeration of every site set with Dijkstra times.

    uv run python examples/synthetic_city/verify_enumeration.py [workspace]   # default workspace/synthetic_city

For every run folder in <workspace>/runs, and P = number of sites in that run, all candidate subsets of size P are
evaluated by multi-source Dijkstra on the built routing graph; the run's objective values must equal the enumerated
optima (Utilitarian sum D*t; Individual Stage 1 max Q4 t and Stage 2 min squared deviation among Stage 1 optima;
District worst Q4 district mean over ALL candidates). Exit code 0 if every proven model matches.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

from childcare_access import evaluation
from childcare_access import solutions as sol
from childcare_access.script_support import read_pickle

TOL = 1e-6


class _NoSites:
    source = Path("synthetic run")

    def get(self, key: str, default: object = None) -> object:
        return {} if key == "reproducibility.reported_sites" else default


def enumerate_all(inputs: Path, P: int) -> pd.DataFrame:
    """Every P-site set of the candidates with its objectives, walking times recomputed by Dijkstra."""
    bundle = read_pickle(inputs / "inputs_bundle.pkl")
    graph = read_pickle(inputs / "walk_graph_layered.pkl")
    demand = bundle["demand"].set_index("demand_id")
    nodes = bundle["candidates"].set_index("facility_id")["nearest_node"]
    kitas, q4, D = (
        bundle["kitas"]["nearest_node"],
        demand["least_advantaged"].astype(bool),
        demand["estimated_demand"],
    )
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
    return pd.DataFrame(rows).set_index("sites")


def main(workspace: Path) -> int:
    """Compare the solved models of a workspace with the exhaustive enumeration; return the exit code."""
    rows, ok = [], True
    for run in sorted((workspace / "runs").iterdir()):
        solved = sol.load_reported_solutions(run_dir=run, config=_NoSites(), require_all=False)
        if not solved:
            continue
        P = len(next(iter(solved.values())).sites)
        e = enumerate_all(workspace / "inputs", P)
        for key, s in solved.items():
            chosen = e.loc[[tuple(sorted(s.sites))]].iloc[0]
            if key == "utilitarian":
                value, optimum = chosen["utilitarian"], e["utilitarian"].min()
            elif key == "district":
                value, optimum = chosen["district"], e["district"].min()
            else:
                z = e["z"].min()
                value, optimum = (
                    (chosen["z"], chosen["stage2"]),
                    (z, e.loc[e["z"] <= z + TOL, "stage2"].min()),
                )
            match = (
                abs(value[0] - optimum[0]) <= TOL * max(1, abs(optimum[0]))
                and abs(value[1] - optimum[1]) <= TOL * max(1, abs(optimum[1]))
                if key == "individual"
                else abs(value - optimum) <= TOL * max(1, abs(optimum))
            )
            ok &= match or s.certification != sol.PROVEN  # only a PROVEN claim must equal the enumerated optimum
            rows.append(
                {
                    "run": run.name,
                    "P": P,
                    "model": key,
                    "certification": s.certification,
                    "sites": " ".join(x.rsplit("_", 1)[-1] for x in s.sites),
                    "objective": value,
                    "enumerated optimum": optimum,
                    "site sets": len(e),
                    "matches": match,
                }
            )
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(
        "ENUMERATION CHECK PASSED"
        if ok
        else "ENUMERATION CHECK FAILED: a proven solution differs from the enumerated optimum"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("workspace/synthetic_city")))
