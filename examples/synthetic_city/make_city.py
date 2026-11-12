"""Generate the raw input files of the synthetic example city ("Testburg") used by config/synthetic_city.toml.

    uv run python examples/synthetic_city/make_city.py            # writes examples/synthetic_city/data/ (git-ignored)

Nothing here comes from Münster. Layout (EPSG:25832): four 300 m x 300 m districts in a 2 x 2 block; a 7 x 7 grid
of footways at 100 m spacing (an Overpass-format snapshot); 16 residential buildings near grid nodes; one existing
Kita in the south-west corner; a census grid (EPSG:3035, 100 m) with children in the cells of the buildings plus one
cell without a building (a fallback origin); forecast and social-indicator tables in the template's long format.
With the configured 50th-percentile rule, Nordost and Suedost form the least-advantaged group.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, box

from childcare_access import (
    adapters,
)  # the files are written through the documented adapters, as a new city would

DATA = Path(__file__).resolve().parent / "data"
X0, Y0, STEP, N = 400_000.0, 5_750_000.0, 100.0, 7
DISTRICTS = {
    1: ("Nordwest", 0, 300),
    2: ("Nordost", 300, 300),
    3: ("Suedwest", 0, 0),
    4: ("Suedost", 300, 0),
}
CHILDREN = {1: (6, 9), 2: (12, 15), 3: (5, 7), 4: (10, 11)}  # aged 0-2 / 3-5 per district, 2030
INDICATORS = {
    1: (4, 6, 20, 10),
    2: (20, 30, 55, 30),
    3: (6, 8, 25, 12),
    4: (15, 25, 50, 25),
}  # four index indicators (%)
LABELS = {
    "pct_welfare_15_64": "Basic income support 15-64 (%)",
    "pct_welfare_0_14": "Basic income support 0-14 (%)",
    "pct_children_migration_background": "Children with migration background (%)",
    "pct_children_single_parent": "Children in single-parent households (%)",
}
EXPECTED_Q4 = ["Nordost", "Suedost"]
SITES = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
    (4, 1),
    (4, 2),
    (5, 1),
    (5, 2),
    (4, 4),
    (4, 5),
    (5, 4),
    (5, 5),
]


def _node(i: int, j: int) -> int:
    return 1000 + i * N + j


def write_data(data: Path = DATA) -> Path:
    """Write every raw file of the synthetic city into ``data``; returns ``data``."""
    data.mkdir(parents=True, exist_ok=True)
    (data / "osm").mkdir(exist_ok=True)
    to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)
    to_3035 = Transformer.from_crs(25832, 3035, always_xy=True)

    polygons = [box(X0 + dx, Y0 + dy, X0 + dx + 300, Y0 + dy + 300) for _, dx, dy in DISTRICTS.values()]
    gpd.GeoDataFrame(
        {"district_id": list(DISTRICTS), "district_name": [v[0] for v in DISTRICTS.values()]},
        geometry=polygons,
        crs=25832,
    ).to_file(data / "districts.gpkg", driver="GPKG")

    elements = []
    for i in range(N):
        for j in range(N):
            lon, lat = to_wgs.transform(X0 + j * STEP, Y0 + i * STEP)
            elements.append({"type": "node", "id": _node(i, j), "lat": lat, "lon": lon})
    for k in range(N):
        elements.append(
            {
                "type": "way",
                "id": 1 + k,
                "nodes": [_node(k, j) for j in range(N)],
                "tags": {"highway": "footway"},
            }
        )
        elements.append(
            {
                "type": "way",
                "id": 1 + N + k,
                "nodes": [_node(i, k) for i in range(N)],
                "tags": {"highway": "footway"},
            }
        )
    (data / "osm" / "highways.json").write_text(
        json.dumps({"version": 0.6, "osm3s": {"timestamp_osm_base": "synthetic"}, "elements": elements}),
        encoding="utf-8",
    )

    footprints = [box(X0 + j * STEP + 8, Y0 + i * STEP + 8, X0 + j * STEP + 20, Y0 + i * STEP + 20) for i, j in SITES]
    source = gpd.GeoDataFrame(
        {
            "ref": [f"b{k:02d}" for k in range(len(SITES))],
            "use": ["mixed"] + ["housing"] * (len(SITES) - 1),
        },
        geometry=footprints,
        crs=25832,
    )
    adapters.buildings_layer(source, residential=True, mixed_use=source["use"].eq("mixed"), id_column="ref").to_file(
        data / "buildings.gpkg", layer="buildings", driver="GPKG"
    )

    cells = {}
    for k, poly in enumerate(footprints):
        x, y = to_3035.transform(poly.centroid.x, poly.centroid.y)
        key = (int(x // 100), int(y // 100))
        cells[key] = cells.get(key, 0) + 2 + (k % 3)
    x, y = to_3035.transform(X0 + 560, Y0 + 40)  # district Suedost, no building
    cells[(int(x // 100), int(y // 100))] = 6
    grid = pd.DataFrame(
        [
            {
                "xc": gx * 100 + 50,
                "yc": gy * 100 + 50,
                "under3": c // 2,
                "three_to_five": c - c // 2,
            }
            for (gx, gy), c in cells.items()
        ]
    )
    adapters.census_grid_zip(
        grid,
        data / "census_grid.zip",
        cell_size_m=100,
        x_column="xc",
        y_column="yc",
        count_columns={"children_0_2": "under3", "children_3_5": "three_to_five"},
    )

    wide = pd.DataFrame(
        {
            "nr": list(DISTRICTS),
            "name": [v[0] for v in DISTRICTS.values()],
            "kids_0_2": [v[0] for v in CHILDREN.values()],
            "kids_3_5": [v[1] for v in CHILDREN.values()],
            **{label: [INDICATORS[nr][i] for nr in DISTRICTS] for i, label in enumerate(LABELS.values())},
        }
    )
    adapters.forecast_long(
        wide, "nr", {"age_0_2": "kids_0_2", "age_3_5": "kids_3_5"}, year=2030, name_column="name"
    ).to_csv(data / "forecast.csv", index=False)
    adapters.indicators_long(wide, "nr", LABELS, date="31.12.2029", name_column="name", year=2029).to_csv(
        data / "social.csv", index=False
    )

    adapters.facilities_file(
        gpd.GeoDataFrame({"name": ["Kita Ecke"]}, geometry=[Point(X0 + 5, Y0 + 5)], crs=25832),
        data / "existing_facilities.geojson",
        name_column="name",
    )
    (data / "student_residence_decisions.csv").write_text("building_id,decision,category\n", encoding="utf-8")
    return data


if __name__ == "__main__":
    print(f"synthetic city data written to {write_data(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA)}")
