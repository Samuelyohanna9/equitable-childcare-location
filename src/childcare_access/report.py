"""Tables and figures of the notebooks (results of the three models).

Presentation only: every number comes from the package (``evaluation``, ``solutions``, ``comparison``) or from the
saved, checked results; nothing here solves a model or writes a file.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

MODELS = {
    "utilitarian": ("Utilitarian", "#2a78d6"),
    "individual": ("Individual Rawlsian", "#eb6834"),
    "district": ("District Rawlsian", "#1baf7a"),
}
# Model-level certification codes (solutions.certify) in words.
CERTIFICATION = {
    "proven_optimal": "proven optimal",
    "optimal_within_requested_gap": "optimal within the requested gap (not proven)",
    "stage1_proven_optimal__stage2_time_limited_not_proven": "Stage 1 proven optimal; Stage 2 time-limited (not "
    "proven)",
    "stage1_proven_optimal__stage2_not_proven": "Stage 1 proven optimal; Stage 2 not proven",
    "exact_stage2_by_enumeration__reference_only": "exact reference by enumeration (not reported)",
    "feasible_not_proven": "feasible, not proven optimal",
    "no_solution": "no solution",
}
AUDIT_COLUMNS = {
    "current": "Current",
    "utilitarian": "Utilitarian",
    "individual": "Individual (Gurobi)",
    "district": "District",
    "individual_exact_stage2": "Individual (exact Stage 2)",
}
# Indicator names of evaluation.accessibility_indicators / savings -> row names of the saved results table
# (``[outputs] audit``).
INDICATOR_ROWS = {
    "City mean": "City mean",
    "City demand-weighted mean": "City dw mean",
    "City median": "City median",
    "City SD": "City SD",
    "City P90": "City P90",
    "City maximum": "City max",
    "Q4 mean": "Q4 mean",
    "Q4 demand-weighted mean": "Q4 dw mean",
    "Q4 median": "Q4 median",
    "Q4 SD": "Q4 SD",
    "Q4 P90": "Q4 P90",
    "Worst Q4 location": "Worst Q4 location",
    "Worst Q4 district mean": "Worst Q4 district mean",
    "SD of Q4 district means": "SD of Q4 district means",
}
GAIN_ROWS = {
    "City child-min saved": "City child-min saved",
    "Q4 child-min saved": "Q4 child-min saved",
    "Children improved (> 0), city": "Children > 0",
    "Children improved (> 0), Q4": "Q4 children > 0",
    "Children improved (>= 5 min), city": "Children >= 5",
    "Children improved (>= 5 min), Q4": "Q4 children >= 5",
}


def indicators(times: dict, demand: pd.DataFrame, group: str = "Q4") -> pd.DataFrame:
    """Accessibility indicators (``evaluation.accessibility_indicators``) per scenario, one column each."""
    from .evaluation import accessibility_indicators

    return pd.DataFrame({key: accessibility_indicators(t, demand, group=group) for key, t in times.items()})


def gains(times: dict, demand: pd.DataFrame) -> pd.DataFrame:
    """Child-minutes saved and children improved against the current situation (``evaluation.savings``)."""
    from .evaluation import savings

    return pd.DataFrame({key: savings(times["current"], t, demand) for key, t in times.items() if key != "current"})


def check_against_audit(table: pd.DataFrame, audit: pd.DataFrame, rows: dict, atol: float = 1e-9) -> float:
    """Largest absolute difference between recomputed values and the audited table; raises above ``atol``."""
    worst = 0.0
    for key, column in AUDIT_COLUMNS.items():
        if key not in table:
            continue
        for ours, theirs in rows.items():
            if ours in table.index and theirs in audit.index:
                worst = max(worst, abs(float(table.at[ours, key]) - float(audit.at[theirs, column])))
    if worst > atol:
        raise AssertionError(f"recomputed results differ from the audited table by {worst:.3g}")
    return worst


def relative_change(table: pd.DataFrame) -> pd.DataFrame:
    """Every numeric indicator relative to the current situation.

    (model value / current value) - 1, one column per model.
    """
    rows = [row for row in INDICATOR_ROWS if row in table.index]
    values = table.loc[rows].astype(float)
    return values.drop(columns="current").div(values["current"], axis=0).sub(1)


def priority_chart(table: pd.DataFrame, group: str = "Q4") -> Figure:
    """Mean and maximum walks of the priority group per scenario.

    One panel per indicator, with the scenarios in the same order.
    """
    keys = [key for key in ("current", *MODELS) if key in table]
    colours = {"current": ("Current", "#8a8985"), **MODELS}
    rows = [
        f"{group} demand-weighted mean",
        f"Worst {group} location",
        f"Worst {group} district mean",
    ]
    fig, axes = plt.subplots(1, len(rows), figsize=(13, 4.6))
    for ax, row in zip(axes, rows, strict=True):
        values = table.loc[row, keys].astype(float)
        bars = ax.bar(
            range(len(keys)),
            values,
            width=0.62,
            color=[colours[key][1] for key in keys],
            alpha=0.85,
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=10)
        ax.set_xticks(range(len(keys)), [colours[key][0].replace(" ", "\n") for key in keys])
        ax.set_title(row, fontsize=11)
        ax.set_ylim(0, values.max() * 1.14)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Walking time (minutes)")
    fig.tight_layout()
    return fig


def changed_walk_boxplot(times: dict[str, pd.Series]) -> Figure:
    """Walking times of the locations whose walk changes in at least one model. Each location counts once."""
    keys = [key for key in ("current", *MODELS) if key in times]
    frame = pd.DataFrame({key: times[key] for key in keys})
    rows = frame[frame.sub(frame["current"], axis=0).abs().max(axis=1) > 1e-9]
    colours = {"current": ("Current", "#8a8985"), **MODELS}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    box = ax.boxplot(
        [rows[key] for key in keys],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
    )
    for patch, key in zip(box["boxes"], keys, strict=True):
        patch.set_facecolor(colours[key][1])
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(keys) + 1), [colours[key][0] for key in keys])
    ax.set_ylabel("Walking time to the nearest Kita (minutes)")
    ax.set_title(
        f"Locations whose walk changes in at least one model (n = {len(rows):,} of {len(frame):,})",
        fontsize=11,
    )
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def distribution_boxplots(
    times: dict[str, pd.Series],
    demand: pd.DataFrame,
    group_column: str = "least_advantaged",
    group: str = "Q4",
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Walking-time distribution per demand location, citywide and in the priority group (one box per scenario).

    Each location counts once (estimated children are not weights). Outlier markers are hidden and ``ylim`` only
    limits the display; every location enters every box, median and mean.
    """
    from matplotlib.lines import Line2D

    scenarios = {"current": ("Current", "#8a8985"), **MODELS}
    panels = {
        "Citywide: all demand locations": pd.Series(True, index=demand.index),
        f"{group}: demand locations in {group} districts": demand[group_column].astype(bool),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (title, selected) in zip(axes, panels.items(), strict=True):
        values = [times[key][selected] for key in scenarios]
        box = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.55,
            whis=1.5,
            showfliers=False,
            showmeans=True,
            medianprops={"color": "black", "linewidth": 1.6},
            meanprops={"marker": "D", "markersize": 6, "markerfacecolor": "white", "markeredgecolor": "black"},
            whiskerprops={"color": "#444444"},
            capprops={"color": "#444444"},
        )
        for patch, (_, colour) in zip(box["boxes"], scenarios.values(), strict=True):
            patch.set_facecolor(colour)
            patch.set_alpha(0.85)
            patch.set_edgecolor("#444444")
        ax.set_xticks(range(1, len(scenarios) + 1), [label.replace(" ", "\n") for label, _ in scenarios.values()])
        ax.set_title(f"{title} (n = {int(selected.sum()):,})", fontsize=11)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Walking time to the nearest Kita (minutes)")
    mean_marker = Line2D(
        [],
        [],
        marker="D",
        linestyle="none",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="black",
        label="Mean (unweighted)",
    )
    fig.legend(
        handles=[Line2D([], [], color="black", linewidth=1.6, label="Median"), mean_marker],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Walking-time distribution per demand location, P = 3", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def status_summary(run: Path) -> pd.DataFrame:
    """One line per model: the certification category of each solver call (``comparison.status_table``)."""
    from .comparison import status_table

    table = status_table(run)
    table["model"] = table["model"].replace({"rawlsian": "individual"})
    return table


# figures
def _base(
    ax: Axes, districts: gpd.GeoDataFrame, groups_column: str = "least_advantaged", title: str = ""
) -> gpd.GeoDataFrame:
    """Draw the districts (priority group shaded) on ``ax``; return them in EPSG:25832."""
    shape = districts.to_crs(25832)
    shape.plot(
        ax=ax,
        color=shape[groups_column].map({True: "#f2c6bd", False: "#f4f4f2"}),
        edgecolor="#9a9a96",
        linewidth=0.4,
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    return shape


def sites_map(
    districts: gpd.GeoDataFrame, kitas: gpd.GeoDataFrame, candidates: gpd.GeoDataFrame, solutions: dict
) -> Figure:
    """Q4 districts, existing Kitas and the sites of each model (one panel per model)."""
    fig, axes = plt.subplots(1, len(solutions), figsize=(5 * len(solutions), 5.2))
    sites = candidates.set_index("facility_id").to_crs(25832)
    for ax, (key, s) in zip(axes, solutions.items(), strict=True):
        name, colour = MODELS[key]
        _base(ax, districts, title=f"{name}\n{CERTIFICATION.get(s.certification, s.certification)}")
        kitas.to_crs(25832).plot(ax=ax, color="#5a5a56", markersize=4)
        sites.loc[list(s.sites)].plot(ax=ax, color=colour, markersize=90, marker="*", edgecolor="black", linewidth=0.5)
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#f2c6bd"),
            plt.Line2D([], [], color="#5a5a56", marker="o", ls=""),
            plt.Line2D([], [], color="black", marker="*", ls="", markersize=10),
        ],
        labels=["Q4 district", "existing Kita", "new site"],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def access_map(
    districts: gpd.GeoDataFrame, demand: gpd.GeoDataFrame, times: pd.Series, title: str, vmax: float = 20
) -> Figure:
    """Walking time to the nearest open facility at every demand location (darker = longer)."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _base(ax, districts, title=title)
    points = demand.to_crs(25832)
    shown = points.assign(t=times.reindex(points["demand_id"]).to_numpy()).sort_values("t")
    shown.plot(
        ax=ax,
        column="t",
        cmap="magma_r",
        vmin=0,
        vmax=vmax,
        markersize=2,
        legend=True,
        legend_kwds={"label": f"walking time (min; colours capped at {vmax:g})", "shrink": 0.6},
    )
    fig.tight_layout()
    return fig


def demand_map(districts: gpd.GeoDataFrame, demand: gpd.GeoDataFrame) -> Figure:
    """Estimated children aged 0-5 per demand location (Q4 districts shaded)."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _base(ax, districts, title="Estimated children aged 0-5 per demand location (2026)")
    points = demand.to_crs(25832).sort_values("estimated_demand")
    points.plot(
        ax=ax,
        column="estimated_demand",
        cmap="Blues",
        vmin=0,
        vmax=float(points["estimated_demand"].quantile(0.99)),
        markersize=2,
        legend=True,
        legend_kwds={"label": "estimated children", "shrink": 0.6},
    )
    fig.tight_layout()
    return fig


def savings_grid(demand: gpd.GeoDataFrame, times: dict, key: str, cell: float = 500) -> gpd.GeoDataFrame:
    """Demand-weighted mean walking-time saving of one model per square grid cell (EPSG:25832).

    Cell value: sum_i D_i (c_i - t_i) / sum_i D_i over all demand locations in the cell, with locations without a
    saving counted as 0. Only cells with demand are returned, so an empty cell never appears as zero improvement.
    """
    from shapely.geometry import box as cell_box

    points = demand.to_crs(25832)
    weight = points["estimated_demand"].to_numpy()
    saving = (times["current"] - times[key]).clip(lower=0).reindex(points["demand_id"]).to_numpy()
    frame = pd.DataFrame(
        {
            "cx": (points.geometry.x // cell).astype(int).to_numpy(),
            "cy": (points.geometry.y // cell).astype(int).to_numpy(),
            "D": weight,
            "DS": weight * saving,
        }
    )
    cells = frame.groupby(["cx", "cy"])[["D", "DS"]].sum().reset_index()
    cells = cells[cells["D"] > 0].assign(saving=lambda c: c["DS"] / c["D"])
    geometry = [
        cell_box(x * cell, y * cell, (x + 1) * cell, (y + 1) * cell)
        for x, y in zip(cells["cx"], cells["cy"], strict=True)
    ]
    return gpd.GeoDataFrame(cells, geometry=geometry, crs=25832)


def savings_grid_maps(
    districts: gpd.GeoDataFrame,
    demand: gpd.GeoDataFrame,
    times: dict,
    candidates: gpd.GeoDataFrame,
    solutions: dict,
    *,
    cell: float = 500,
    groups_column: str = "least_advantaged",
    group: str = "Q4",
) -> Figure:
    """Walking-time savings of each model on one regular grid (same cells, weights, classes and map extent).

    Blue: demand-weighted mean minutes saved per cell (``savings_grid``). Light grey: cells with demand but no saving.
    Cells without demand are not drawn. Stars: the model's new sites. The extent contains every cell that gains in any
    model, with a 1.5 km margin.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    keys = [key for key in MODELS if key in solutions]
    grids = {key: savings_grid(demand, times, key, cell) for key in keys}
    reference = grids[keys[0]].set_index(["cx", "cy"])["D"]
    assert all(g.set_index(["cx", "cy"])["D"].equals(reference) for g in grids.values())  # same cells and weights
    top = max(float(g["saving"].max()) for g in grids.values())
    classes = [0, 2, 4, 8, 12, max(16, float(int(top) + 1))]
    cmap = ListedColormap(["#c6dbef", "#6baed6", "#3182bd", "#08519c", "#08306b"])
    norm = BoundaryNorm(classes, cmap.N)
    shape = districts.to_crs(25832)
    sites = candidates.set_index("facility_id").to_crs(25832)
    gained = pd.concat([g[g["saving"] > 1e-9] for g in grids.values()])
    x0, y0, x1, y1 = gained.total_bounds
    pad = 1500
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 6.0))
    for ax, key in zip(axes, keys, strict=True):
        cells = grids[key]
        has_gain = cells["saving"] > 1e-9
        shape.plot(
            ax=ax,
            color=shape[groups_column].map({True: "#fbebe7", False: "white"}),
            edgecolor="#cfcfcb",
            linewidth=0.4,
        )
        cells[~has_gain].plot(ax=ax, color="#e6e6e3", alpha=0.7, linewidth=0)
        shape[shape[groups_column]].boundary.plot(ax=ax, color="#c96b66", linewidth=0.6)
        cells[has_gain].plot(ax=ax, column="saving", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.4)
        sites.loc[list(solutions[key].sites)].plot(
            ax=ax, marker="*", markersize=110, color="#f5b800", edgecolor="black", linewidth=0.7, zorder=5
        )
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_title(MODELS[key][0], fontsize=11)
        ax.set_axis_off()
    handles = [Patch(facecolor=cmap(i), label=f"{lo:g}\u2013{hi:g}") for i, (lo, hi) in enumerate(pairwise(classes))]
    handles += [
        Patch(facecolor="#e6e6e3", label="no saving"),
        Patch(facecolor="#fbebe7", edgecolor="#c96b66", label=f"{group} district"),
        plt.Line2D(
            [], [], marker="*", color="#f5b800", markeredgecolor="black", markersize=12, ls="", label="new Kita"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=9,
        title="Minutes saved",
        title_fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig
