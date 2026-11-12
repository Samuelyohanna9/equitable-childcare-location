"""Evaluation of facility configurations, independent of the optimisation models.

Walking times are recomputed by one multi-source Dijkstra on the reversed routing graph from every open facility
(existing facilities plus the selected sites): each demand location gets the time to its nearest open facility.
The indicators and savings below are those of the results tables (location statistics unweighted,
population SD, linear percentiles; district means weighted by estimated demand).

``demand`` is a table indexed by demand location with the columns ``estimated_demand``, ``least_advantaged``,
``district_name``, ``vulnerability_quartile`` and ``nearest_node`` (the demand table of the inputs bundle,
indexed by ``demand_id``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx
import pandas as pd

from .all_demand_candidates import validate_selected_solution_with_dijkstra  # noqa: F401  (re-exported)
from .core import DISTRICT_NAME


def nearest_open_times(
    graph: nx.DiGraph, demand_nodes: pd.Series, facility_nodes: Iterable[int], weight: str = "walk_time_min"
) -> pd.Series:
    """Walking time from every demand node to its nearest open facility (index of ``demand_nodes`` kept)."""
    reverse = graph.reverse(copy=False) if graph.is_directed() else graph
    dist = nx.multi_source_dijkstra_path_length(reverse, {int(n) for n in facility_nodes}, weight=weight)
    return demand_nodes.astype("int64").map(dist).astype(float)


def scenario_times(
    graph: nx.DiGraph,
    demand: pd.DataFrame,
    existing_nodes: Iterable[int],
    candidate_nodes: pd.Series,
    sites: Sequence[str] = (),
) -> pd.Series:
    """Nearest-open walking times with the existing facilities plus the candidate ``sites`` open.

    ``candidate_nodes`` maps candidate ID -> network node.
    """
    return nearest_open_times(
        graph,
        demand["nearest_node"],
        set(map(int, existing_nodes)) | {int(candidate_nodes[s]) for s in sites},
    )


def district_means(times: pd.Series, demand: pd.DataFrame, least_advantaged_only: bool = True) -> pd.Series:
    """Demand-weighted mean walking time per district (by default the least-advantaged districts only)."""
    d = demand[demand["least_advantaged"].astype(bool)] if least_advantaged_only else demand
    weight = d["estimated_demand"]
    return (times[d.index] * weight).groupby(d[DISTRICT_NAME]).sum() / weight.groupby(d[DISTRICT_NAME]).sum()


def _named(values: dict, group: str) -> dict:
    """Indicator keys name the priority group ("Q4" for the default rule; see vulnerability_modes.group_name)."""
    return values if group == "Q4" else {k.replace("Q4", group): v for k, v in values.items()}


def accessibility_indicators(times: pd.Series, demand: pd.DataFrame, group: str = "Q4") -> dict[str, object]:
    """The accessibility indicators of the model comparison table for one scenario.

    ``group`` is the name of the priority group in the keys.
    """
    q4 = demand["least_advantaged"].astype(bool)
    D = demand["estimated_demand"]
    means = district_means(times, demand)
    return _named(
        {
            "City mean": float(times.mean()),
            "City demand-weighted mean": float((times * D).sum() / D.sum()),
            "City median": float(times.median()),
            "City SD": float(times.std(ddof=0)),
            "City P90": float(times.quantile(0.9)),
            "City maximum": float(times.max()),
            "Q4 mean": float(times[q4].mean()),
            "Q4 demand-weighted mean": float((times[q4] * D[q4]).sum() / D[q4].sum()),
            "Q4 median": float(times[q4].median()),
            "Q4 SD": float(times[q4].std(ddof=0)),
            "Q4 P90": float(times[q4].quantile(0.9)),
            "Worst Q4 location": float(times[q4].max()),
            "Worst Q4 district mean": float(means.max()),
            "Worst Q4 district": means.idxmax(),
            "SD of Q4 district means": float(means.std(ddof=0)),
        },
        group,
    )


def savings(
    current: pd.Series,
    new: pd.Series,
    demand: pd.DataFrame,
    thresholds: Sequence[float] = (1, 2, 5),
    group: str = "Q4",
) -> dict[str, float]:
    """Child-minutes saved and beneficiaries of a scenario relative to the current situation.

    A location benefits when its walking time falls (gain > 0); child-minutes = gain x estimated children. Savings
    are also split by ``vulnerability_quartile``: Q1-Q4 for an index, ``priority``/``other`` for a supplied list.
    """
    gain = current - new
    if (gain < -1e-9).any():
        raise ValueError(
            f"{int((gain < -1e-9).sum())} demand location(s) are worse off than today; opening facilities "
            "cannot lengthen nearest-open walks, so the times are inconsistent"
        )
    D, q4 = demand["estimated_demand"], demand["least_advantaged"].astype(bool)
    saved = gain.clip(lower=0) * D
    out = {
        "City child-min saved": float(saved.sum()),
        "Q4 child-min saved": float(saved[q4].sum()),
        "Children improved (> 0), city": float(D[gain > 0].sum()),
        "Children improved (> 0), Q4": float(D[(gain > 0) & q4].sum()),
    }
    for k in thresholds:
        out[f"Children improved (>= {k} min), city"] = float(D[gain >= k].sum())
        out[f"Children improved (>= {k} min), Q4"] = float(D[(gain >= k) & q4].sum())
    out["Locations improved"] = float((gain > 0).sum())
    out["Largest single gain (min)"] = float(gain.max())
    for quartile in sorted(demand["vulnerability_quartile"].dropna().unique()):
        out[f"Savings {quartile.split()[0]}"] = float(saved[demand["vulnerability_quartile"] == quartile].sum())
    return _named(out, group)


def vulnerability_summary(demand: pd.DataFrame) -> dict[str, object]:
    """Index-dependent figures, or "unavailable" when the priority group is a supplied list without an index.

    The index is a district value inherited by every location, not a measurement of each child.
    """
    index = demand["childcare_vulnerability_index"]
    if index.isna().all():
        return {
            "Vulnerability index": "unavailable (priority group supplied without an index)",
            "Rawls access multiplier": "unavailable (no index)",
        }
    D = demand["estimated_demand"]
    known = index.notna()
    return {
        "Vulnerability index, demand-weighted mean (district values)": float(
            (index[known] * D[known]).sum() / D[known].sum()
        ),
        "Demand in districts without an index value": float(D[~known].sum()),
    }


def lexicographic_stage2_objective(times: pd.Series, demand: pd.DataFrame) -> float:
    """Individual Rawlsian Stage 2 objective: sum over least-advantaged locations of (t_i - their mean)^2."""
    t = times[demand["least_advantaged"].astype(bool)]
    return float(((t - t.mean()) ** 2).sum())
