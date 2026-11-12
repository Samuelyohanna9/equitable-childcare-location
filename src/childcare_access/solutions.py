"""Explicit selection of saved optimisation solutions, with status, certification and provenance.

Two selections, never mixed:
- :func:`load_reported_solutions` - the reported solutions, one per model (for Münster, those of the thesis):
  ``utilitarian``, ``individual`` (the Gurobi two-stage result as saved by the run) and ``district``;
- :func:`load_alternative_solutions` - other saved solutions, clearly labelled and not part of the
  reported results (``individual_exact_stage2``: the exact Stage 2 optimum found by enumeration).

Certification is derived from the literal solver records, never assumed:
- ``proven_optimal``: status Optimal and a final relative gap of at most 1e-6 (``EXACT_GAP``);
- ``optimal_within_requested_gap``: status Optimal with a larger gap, i.e. optimal only within the MIP gap that was
  requested (e.g. 1% for the Utilitarian model); the objective is NOT proven exactly optimal;
- ``stage1_proven_optimal__stage2_time_limited_not_proven``: lexicographic model whose Stage 1 optimum is proven
  but whose Stage 2 stopped at its time limit, so the reported dispersion is NOT proven minimal
  (``stage1_proven_optimal__stage2_not_proven`` if Stage 2 stopped for another reason);
- ``exact_stage2_by_enumeration__reference_only``: Stage 2 optimum among all enumerated Stage 1 optima (no solver);
- ``feasible_not_proven``: a feasible solution without any of the proofs above;
- ``no_solution``: the solver found no solution.

If the configuration lists the reported sites (``[reproducibility.reported_sites]``), the saved solutions must
contain exactly those sites; otherwise :class:`SolutionSelectionError` is raised, so a reported facility location is
never replaced silently.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .config import CityConfig

from .core import LocationModelResult
from .settings import CONFIG

PROVEN = "proven_optimal"
WITHIN_GAP = "optimal_within_requested_gap"
STAGE2_TIME_LIMITED = "stage1_proven_optimal__stage2_time_limited_not_proven"
STAGE2_NOT_PROVEN = "stage1_proven_optimal__stage2_not_proven"
EXACT_REFERENCE = "exact_stage2_by_enumeration__reference_only"
FEASIBLE = "feasible_not_proven"
NO_SOLUTION = "no_solution"
EXACT_GAP = 1e-6

# Model key -> (result folder, solver-record roles). Role names are those written by the runner.
MODELS = {
    "utilitarian": ("utilitarian", ["Utilitarian"]),
    "individual": ("rawlsian", ["Individual Rawlsian Stage 1", "Individual Rawlsian Stage 2"]),
    "district": ("district", ["District Rawlsian (screened)"]),
}
LABELS = {
    "utilitarian": "Utilitarian",
    "individual": "Individual Rawlsian",
    "district": "District Rawlsian",
}


class SolutionSelectionError(RuntimeError):
    """Saved solutions are inconsistent with their records or with the reported sites of the configuration."""


@dataclass(frozen=True)
class Solution:
    """One saved solution: its sites, objective, status and certification, and whether the thesis reports it."""

    key: str
    label: str
    reported_in_thesis: bool
    sites: tuple[str, ...]
    certification: str
    status: str  # literal status of the saved result
    stage_statuses: dict  # literal solver status per stage/role
    objective_value: float
    secondary_objective_value: float | None
    provenance: dict
    result: LocationModelResult = field(repr=False, compare=False)

    @property
    def proven_optimal(self) -> bool:
        """True only if every stage was proven optimal by the solver."""
        return self.certification == PROVEN


def _sha256(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(run: Path) -> pd.DataFrame:
    """All solver records of a run folder, one row per solver call."""
    frames = [pd.read_csv(f) for f in sorted(run.glob("*/*_solver_records.csv"))]
    if not frames:
        raise SolutionSelectionError(f"{run}: no solver records found")
    records = pd.concat(frames, ignore_index=True)
    records["role"] = records["role"].str.replace(r" \(attempt \d+\)$", "", regex=True)
    return records.drop_duplicates("role", keep="last").set_index("role")


def certify_stage(record: Mapping[str, Any]) -> str:
    """Certification of one solver run from its record (status, mip_gap, and incumbent if any)."""
    gap = record.get("mip_gap")
    gap = None if gap is None or pd.isna(gap) else float(gap)
    incumbent = record.get("incumbent_objective")
    if incumbent is None or pd.isna(incumbent):
        return NO_SOLUTION
    if record.get("status") == "Optimal":
        return PROVEN if gap is not None and gap <= EXACT_GAP else WITHIN_GAP
    return FEASIBLE


def certify(records: Sequence[Mapping[str, Any]]) -> str:
    """Certification of a model from the records of its stages (one, or two for the lexicographic model)."""
    stages = [certify_stage(r) for r in records]
    if NO_SOLUTION in stages:
        return NO_SOLUTION
    if len(stages) == 1 or all(s == PROVEN for s in stages):
        return stages[0] if len(stages) == 1 else PROVEN
    if stages[0] == PROVEN:
        return STAGE2_TIME_LIMITED if records[1].get("status") == "TimeLimit" else STAGE2_NOT_PROVEN
    return FEASIBLE


def _load(run: Path, folder: str, stem: str) -> tuple[LocationModelResult, Path]:
    """The pickled result of one model and its path."""
    path = run / folder / f"{stem}_result.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle), path


def load_reported_solutions(
    run_dir: str | Path | None = None, config: CityConfig = CONFIG, require_all: bool = True
) -> dict[str, Solution]:
    """The reported solution of each model, checked against its solver records and the configured sites.

    ``require_all=False`` (fresh runs of a subset of the models) skips models without a saved result.
    """
    run = Path(run_dir or config.get("outputs.final_run"))
    records = _records(run)
    manifests = sorted(run.glob("run_manifest_*.json"))
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8")) if manifests else {}
    expected = config.get("reproducibility.reported_sites", {})
    note = config.get("reproducibility.individual_note", None)
    out = {}
    for key, (folder, roles) in MODELS.items():
        if not require_all and not (run / folder / f"{folder}_result.pkl").exists():
            continue
        result, path = _load(run, folder, folder)
        missing = [r for r in roles if r not in records.index]
        if missing:
            raise SolutionSelectionError(f"{run}: solver records lack {missing} for the {LABELS[key]} model")
        rows = [records.loc[r] for r in roles]
        sites = tuple(sorted(result.selected_candidate_ids))
        recorded = tuple(sorted(rows[-1]["selected_candidate_ids"].split(" | ")))
        if sites != recorded:
            raise SolutionSelectionError(
                f"{LABELS[key]}: saved result sites {sites} differ from the solver record {recorded}"
            )
        if key in expected and sites != tuple(sorted(expected[key])):
            raise SolutionSelectionError(
                f"{LABELS[key]}: the saved solution {sites} is not the reported solution "
                f"{tuple(sorted(expected[key]))} of {config.source.name}; refusing to replace it"
            )
        provenance = {
            "run_dir": run.as_posix(),
            "result_file": path.as_posix(),
            "result_sha256": _sha256(path),
            "solver_record_roles": roles,
            "gurobi_version": manifest.get("gurobi_version"),
            "run_started_utc": manifest.get("started_utc"),
            "selection": "reported",
        }
        if key == "individual":
            provenance["stage1_sites"] = sorted(rows[0]["selected_candidate_ids"].split(" | "))
            provenance["rule"] = (
                "Gurobi two-stage result as saved by the run: the Stage 2 incumbent, which equals the "
                "Stage 1 solution when Stage 2 does not improve on its start"
            )
            if note:
                provenance["decision"] = note
        out[key] = Solution(
            key=key,
            label=LABELS[key],
            reported_in_thesis=True,
            sites=sites,
            certification=certify(rows),
            status=str(result.status),
            stage_statuses={r: str(records.loc[r, "status"]) for r in roles},
            objective_value=float(result.objective_value),
            secondary_objective_value=None
            if result.secondary_objective_value is None
            else float(result.secondary_objective_value),
            provenance=provenance,
            result=result,
        )
    return out


def load_alternative_solutions(run_dir: str | Path | None = None, config: CityConfig = CONFIG) -> dict[str, Solution]:
    """Saved solutions that are NOT part of the reported results, each clearly labelled."""
    run = Path(run_dir or config.get("outputs.final_run"))
    out = {}
    exact_dir = run / "rawlsian_exact_stage2"
    if exact_dir.is_dir():
        result, path = _load(run, "rawlsian_exact_stage2", "rawlsian")
        details = json.loads((exact_dir / "rawlsian_result.json").read_text(encoding="utf-8"))
        out["individual_exact_stage2"] = Solution(
            key="individual_exact_stage2",
            label="Individual Rawlsian - exact Stage 2 (reference, not reported)",
            reported_in_thesis=False,
            sites=tuple(sorted(result.selected_candidate_ids)),
            certification=EXACT_REFERENCE,
            status=str(details.get("status", result.status)),
            stage_statuses={"Stage 1": "Optimal (Gurobi)", "Stage 2": "exact enumeration"},
            objective_value=float(result.objective_value),
            secondary_objective_value=None
            if result.secondary_objective_value is None
            else float(result.secondary_objective_value),
            provenance={
                "run_dir": run.as_posix(),
                "result_file": path.as_posix(),
                "result_sha256": _sha256(path),
                "method": details.get("method"),
                "stage1_optimal_triples": details.get("stage1_optimal_triples"),
                "selection": "alternative (not reported)",
            },
            result=result,
        )
    return out
