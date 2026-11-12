"""The district objective: the demand-adjusted mean walking time of each Q4 district, in minutes."""

import geopandas as gpd
from shapely.geometry import Point

from childcare_access.district_demand_maximin import q4_district_demand_adjusted_accessibility


def _assignment() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "demand_id": [1, 2, 3, 4],
            "district_name": ["A", "A", "B", "B"],
            "least_advantaged": [True, True, True, False],
            "childcare_vulnerability_index": [0.9, 0.9, 0.8, 0.2],
            "estimated_demand": [2.0, 8.0, 10.0, 5.0],
            "walk_time_min": [30.0, 20.0, 10.0, 5.0],
        },
        geometry=[Point(7.6, 51.9)] * 4,
        crs=4326,
    )


def test_q4_district_metric_is_demand_adjusted_and_keeps_minutes() -> None:
    result = q4_district_demand_adjusted_accessibility(_assignment())
    values = result.set_index("district_name")

    # A_d(A) = (2*30 + 8*20) / (2 + 8) = 22 minutes.
    assert values.loc["A", "estimated_demand_adjusted_mean_walk_time"] == 22.0
    # District B has only one Q4 location, so its adjusted mean is 10 minutes.
    assert values.loc["B", "estimated_demand_adjusted_mean_walk_time"] == 10.0
