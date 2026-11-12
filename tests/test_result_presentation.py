"""Result tables and figures of the notebooks: percentage change and walking-time box plots."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from childcare_access import report


def test_relative_change_covers_every_numeric_indicator():
    table = pd.DataFrame(
        {"current": [10.0, 4.0, "A"], "utilitarian": [9.0, 4.0, "B"]},
        index=["City mean", "Q4 SD", "Worst Q4 district"],
    )
    change = report.relative_change(table)
    assert list(change.index) == ["City mean", "Q4 SD"] and list(change.columns) == ["utilitarian"]
    assert change.at["City mean", "utilitarian"] == pytest.approx(-0.1) and change.at["Q4 SD", "utilitarian"] == 0


def test_priority_chart_shows_the_mean_and_both_maximum_measures():
    rows = ["Q4 demand-weighted mean", "Worst Q4 location", "Worst Q4 district mean"]
    table = pd.DataFrame(
        {
            "current": [5.0, 24.0, 8.0],
            "utilitarian": [4.0, 21.0, 6.0],
            "individual": [4.5, 18.0, 7.0],
            "district": [3.9, 21.0, 5.0],
            "individual_exact_stage2": [1.0, 1.0, 1.0],
        },
        index=rows,
    )
    fig = report.priority_chart(table)
    assert [ax.get_title() for ax in fig.axes] == rows
    assert [t.get_text() for t in fig.axes[1].get_xticklabels()] == [
        "Current",
        "Utilitarian",
        "Individual\nRawlsian",
        "District\nRawlsian",
    ]
    assert [round(p.get_height(), 1) for p in fig.axes[1].patches] == [24.0, 21.0, 18.0, 21.0]


def test_box_plot_keeps_only_locations_whose_walk_changes():
    times = {
        "current": pd.Series([5.0, 6, 7, 8], index=list("abcd")),
        "utilitarian": pd.Series([4.0, 6, 7, 8], index=list("abcd")),
        "individual": pd.Series([5.0, 3, 7, 8], index=list("abcd")),
        "district": pd.Series([5.0, 6, 7, 2], index=list("abcd")),
        "individual_exact_stage2": pd.Series([1.0, 1, 1, 1], index=list("abcd")),
    }
    fig = report.changed_walk_boxplot(times)
    assert fig.axes[0].get_title() == "Locations whose walk changes in at least one model (n = 3 of 4)"


def _four_scenarios(index, values):
    keys = ["current", "utilitarian", "individual", "district"]
    return {k: pd.Series(v, index=index, dtype=float) for k, v in zip(keys, values, strict=True)}


def test_distribution_boxplots_use_every_location_citywide_and_in_the_priority_group():
    index = list("abcd")
    demand = pd.DataFrame({"least_advantaged": [True, True, False, False]}, index=index)
    times = _four_scenarios(index, [[5, 6, 7, 30], [4, 6, 7, 30], [5, 3, 7, 30], [5, 6, 2, 30]])
    fig = report.distribution_boxplots(times, demand, ylim=(2, 9))
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["Citywide: all demand locations (n = 4)", "Q4: demand locations in Q4 districts (n = 2)"]
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim() == (2, 9)  # one scale; display limit only


def test_savings_grid_is_a_demand_weighted_mean_and_skips_cells_without_demand():
    import geopandas as gpd
    from shapely.geometry import Point

    ids = ["a", "b", "c"]
    demand = gpd.GeoDataFrame(
        {"demand_id": ids, "estimated_demand": [1.0, 3.0, 2.0]},
        geometry=[Point(100, 100), Point(400, 100), Point(1600, 100)],
        crs=25832,
    )
    times = {"current": pd.Series([10.0, 10, 10], index=ids), "utilitarian": pd.Series([2.0, 10, 10], index=ids)}
    cells = report.savings_grid(demand, times, "utilitarian", cell=500).set_index(["cx", "cy"])
    assert list(cells.index) == [(0, 0), (3, 0)]  # the empty cells between them are not returned
    assert cells.at[(0, 0), "saving"] == pytest.approx(8 * 1 / 4)  # (1 x 8 + 3 x 0) / (1 + 3)
    assert cells.at[(3, 0), "saving"] == 0  # demand but no saving
    assert cells["DS"].sum() == pytest.approx(8.0)  # the cells keep the child-minutes saved
