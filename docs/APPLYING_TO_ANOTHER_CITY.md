# Running the analysis for your city

This guide explains how to run the analysis for any city. You do not need to change any code: everything that
differs between cities (file paths, column names, coordinate system, years) goes into one settings file,
`config/<city>.toml`.

A complete worked example is included: "Testburg", a small invented city with four districts
(`examples/synthetic_city/` and `config/synthetic_city.toml`). Run it first to see every step work.

## 1. What you provide and what the tool does for you

| Part | Who handles it |
|---|---|
| The three optimisation models and the Gurobi calls | The tool (`optimisation`, `all_demand_candidates`, `district_demand_maximin`, `solutions`) |
| Walking times to the nearest open facility and the access indicators | The tool (`evaluation`, `validation`) |
| The walking network and its pedestrian-access rules | The tool (`pedestrian_network`), built from an OpenStreetMap snapshot you supply |
| Placing children in buildings, demand locations, candidate sites | The tool (`demand`, `optimisation`) |
| The vulnerability index and the Q4 group | The tool; you can choose the quantile |
| Checks, build, solve and comparison | The tool (`childcare-access check/build/solve/compare`) |
| File paths, coordinate system, column names, labels, codes, years | You, in `config/<city>.toml` |
| Input files in a different layout | You, using the converters in `childcare_access.adapters` (section 3) |
| Equivalent indicators, district boundaries, census grid, network snapshot | You collect them, and record their sources in the settings file |

**Built into the method.** These parts are written into the code and are not settings:
- the three objectives;
- each child walks to the nearest open facility;
- the vulnerability index (four indicators, scaled min–max, equal weights);
- Q4 = districts whose index is at or above the chosen quantile;
- children are placed in buildings using the census grid;
- the age groups 0–2 and 3–5.

Changing any of them is a change of method, not of settings. Results would then no longer be comparable with those
in `docs/METHODS.md`.

**Settings you can change** (in the `[model_parameters]` section; any value you leave out keeps its default):
- the number of new facilities, P;
- the walking speed;
- the Q4 quantile;
- solver time limits, MIP gaps and the Stage 2 tolerances.

Memory and thread limits for Gurobi go in `[solver_resources]`.

## 2. Steps

The easiest route is the notebook `custom_city_analysis.ipynb`. It asks for each input, checks it straight away and
writes the settings file for you, starting from `config/city_template.toml`.

The same steps on the command line:

```bash
cp config/city_template.toml config/<city>.toml                  # 1. fill in every "<-- REPLACE"
uv run childcare-access check --config config/<city>.toml        # 2. checks files, columns, CRS, settings
uv run childcare-access build --config config/<city>.toml        # 3. builds the inputs (offline): workspace/<city>/inputs
uv run childcare-access solve --config config/<city>.toml --p 3  # 4. solves with Gurobi: workspace/<city>/runs/P3
uv run childcare-access compare --config config/<city>.toml --run-name P3_repeat \
    --reference-run workspace/<city>/runs/P3 --reference-inputs workspace/<city>/inputs   # 5. optional
```

**Read the build report before you use the results.** `build` stops at the first check that fails. When it
succeeds, it writes `validation_report.json`, which records the settings, the checksums of your input files, the
package versions, the indicator labels and definitions, and the result of every check:
- the children of each district are all accounted for;
- the walking network is connected;
- the candidate sites are valid;
- the models can be solved.

For a small city you can also check the solutions against every possible choice of sites, as
`examples/synthetic_city/verify_enumeration.py` does.

**Worked example (Testburg).**

```bash
uv run python examples/synthetic_city/make_city.py
uv run childcare-access check --city synthetic_city
uv run childcare-access build --city synthetic_city
uv run childcare-access solve --city synthetic_city --p 3 --run-name P3   # also --p 1, 2 and 4
uv run python examples/synthetic_city/verify_enumeration.py
```

## 3. Input data and converters

| Input | What it must contain | If your file looks different |
|---|---|---|
| District boundaries | One polygon per district, with an ID, a name and a coordinate system | Any vector format works. Set `id_column`, `name_column` and `id_width` |
| Child forecast | Children aged 0–2 and 3–5 per district for one year, as a long table (area label with district code, year, sex, age group, value) | One row per district → `adapters.forecast_long` |
| Social indicators | The four index indicators per district, in %, as a long table; dates end with the year | One row per district → `adapters.indicators_long` (section 4) |
| Building footprints | Polygons with a unique ID and a residential / mixed-use classification | You do not need ALKIS. Any footprints work (national cadastre, OSM, Microsoft or Google buildings) → `adapters.buildings_layer`, using the source's own building-use classes |
| Census grid | Children aged 0–2 and 3–5 per regular grid cell (cell-centre coordinates in a projected CRS) | Any regular grid → `adapters.census_grid_zip`; set `[sources.census] crs` and `cell_size_m` |
| Walking network | Overpass JSON of all `highway=*` ways (with tags) inside the city boundary, at a fixed date | Download it once (section 5). The build itself never downloads anything |
| Existing facilities | Points or polygons of the existing childcare facilities | → `adapters.facilities_file`, then set `osm.kita_file` (or use OSM tags through the OSMnx cache) |
| Student-residence decisions | `building_id, decision (exclude/keep), category` | The file may be empty. With `osm.building_tags_source = "none"` no OSM check of student residences runs |

The converters only reshape and rename columns. They never estimate, fill in or replace values.

## 4. Choosing the vulnerability indicators

The index needs **exactly four** percentages per district, one for each of these concepts (the Münster sources are
given for reference):

| Index column | Concept | Münster indicator |
|---|---|---|
| `pct_welfare_15_64` | Working-age poverty: share of residents aged 15–64 receiving means-tested basic income support | SGB II, Kernindikator 1 |
| `pct_welfare_0_14` | Child poverty: share of children aged 0–14 receiving means-tested basic income support | SGB II, Kernindikator 2 |
| `pct_children_migration_background` | Possible access barriers: share of children with a migration history | Kernindikator 3 (register-based "Migrationsvorgeschichte") |
| `pct_children_single_parent` | Care constraints: share of children in single-parent households | Kontextindikator 1 |

Rules:
- **Same concept, same unit.** Each value is a percentage per district, where higher means more vulnerable. A
  national equivalent is fine if it measures the same concept, for example another country's means-tested
  minimum-income benefit. Record its source, exact definition and any difference in
  `[sources.social.indicator_definitions]`; `check` tells you if this is missing.
- **No stand-ins.** Do not replace a missing indicator with a different one (for example unemployment instead of
  child poverty). If one of the four concepts is not available for your city, the index cannot be computed as
  defined. A different set of indicators is a change of method: explain why you chose it and test how much the
  results depend on it.
- **Recorded as given.** The indicator labels and definitions are copied into the build report unchanged.

## 5. Getting the source data

- **Walking network:** download all `highway=*` ways, with their tags, inside a slightly enlarged city boundary at
  a fixed date, and save the Overpass JSON:

  ```
  [out:json][date:"<snapshot>"]; (way["highway"](poly:"<lat lon ...>");>;); out meta;
  ```

  Record the date in `osm.snapshot_timestamp`. Section 3 of `custom_city_analysis.ipynb` can do this download for
  you (`childcare_access.osm_download`: in tiles, resumable, and it refuses answers older than the date you asked
  for).
- **Existing facilities:** a local file is the simplest source and works offline. The OSM route
  (`amenity=kindergarten`) needs a filled OSMnx cache.
- **Coordinate system:** use a projected CRS in metres that covers the city, for example the local UTM zone.
  `check` rejects geographic and non-metric systems.

## 6. Known limits

- **Census grid.** Children are placed in buildings using gridded child counts (the German Census 2022 has a 100 m
  grid). Without such a grid the method cannot be applied as defined; other population grids or a
  footprint-only allocation would be a change of method.
- **File formats.** The tool expects long tables, a building layer with use codes and a zipped grid CSV. The
  converters cover the common alternatives.
- **Walking rules.** The pedestrian-access rules read OSM tags the way they are used in Germany (for example
  sidewalk tagging). Check them against local tagging practice before you rely on the results.
- **Expected totals.** The `[validation]` totals are optional and specific to each city.
- **Testing so far.** The workflow has been run end to end on the invented city and on Münster, but not yet on a
  second real city.
