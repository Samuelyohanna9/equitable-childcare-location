# childcare-access

Where should a city open new childcare facilities (Kitas) so that children can walk to them? This repository answers
that question for any city with three facility-location models and compares an efficient answer with two fair ones.
It is the code of the MSc thesis *Beyond Efficiency: A Rawlsian Approach to Facility Location for Equitable Access to
Urban Opportunities* (Samuel Yohanna supervised by Dr. Filippe Oliveira Feitosa) with Münster, Germany, as the worked example.

The models choose **P** new sites (default 3) from candidate sites, and every child walks to the nearest open facility:

- **Utilitarian:** minimise the total walking time of all children (efficiency).
- **Individual Rawlsian:** minimise the longest walk of any location in the most vulnerable quarter of districts (Q4),
  then the spread of Q4 walking times. Every Q4 location counts the same, whatever its number of children.
- **District Rawlsian:** minimise the largest child-weighted mean walking time of a Q4 district.

Demand is the *estimated* number of children aged 0-5 at each residential location (district totals spread over a
census grid and building footprints), walking times come from an OpenStreetMap pedestrian network, and Q4 is the
quarter of districts with the highest childcare-vulnerability index. `docs/METHODS.md` describes the method.

## Quick start: the demo city

You need [uv](https://docs.astral.sh/uv/) and a [Gurobi](https://www.gurobi.com/) licence (free for academics; the
demo city is small enough for the licence-free pip version of Gurobi). It works on Windows, macOS and Linux, with
Python 3.12 to 3.14 (uv installs Python for you). On macOS every package installs ready-built on Apple silicon with
macOS 14 or newer and on Intel with macOS 13 or newer.

```bash
git clone https://github.com/Samuelyohanna9/equitable-childcare-location.git childcare-access
cd childcare-access
uv sync --locked                                    # Python environment from uv.lock (includes Gurobi's package)
uv run --locked jupyter lab quickstart.ipynb
```

In the notebook choose **Run → Run All Cells**, type `1` (demo city), press Enter for P = 3, and choose how Gurobi finds
your licence. The demo city "Testburg" is created automatically, its inputs are built and the three models are
solved in about a minute. Results appear at the end of the notebook and are saved in `workspace/guided/demo/`.

## Your own city

Open `custom_city_analysis.ipynb` and run it one section at a time. Each section asks for one input, checks it at
once and saves your answer; at the end it writes the settings file `config/<city>.toml`, builds the inputs, solves
the models and shows the results. It can download the walking network from OpenStreetMap for you. (The other
notebook, `quickstart.ipynb`, does the same through typed questions.)

Have these files ready:

| Needed | Format | Notes |
| --- | --- | --- |
| Districts | map file (`.gpkg`, `.shp`, `.geojson`) with a coordinate system | an ID and a name per district |
| Walking network | OpenStreetMap paths as Overpass JSON | downloaded by the notebook if you have none |
| Existing childcare | map file of points or shapes | |
| Where children live | one of: (a) children per district (CSV) + a census grid + building footprints (the default, used for Münster); (b) children per district + building footprints; (c) known child locations with counts | in Germany the Zensus 2022 grid covers every city (the notebook links it) |
| Priority group | one of: four indicators per district (CSV), from which the tool builds the vulnerability index (the default); a documented index per district; a fixed list of priority districts | the four indicators: welfare recipients aged 15-64 and 0-14, children with a migration background, children in single-parent households |

New facilities can go at residential buildings (the default) or at sites you list in a file. If you choose an option
other than the default, the results mean something different; `docs/APPLYING_TO_ANOTHER_CITY.md` explains what you
provide, how to choose the indicators, and the limits. The walking rules read OpenStreetMap tags as used in Germany, so another
country may need a review of them.

**Time and memory.** Building the inputs needs neither Gurobi nor the internet. A city the size of Münster (11,000
demand locations, 18,000 candidate sites) takes about 30 minutes to build, 0.5-2 hours to solve and about 10 GB of
memory. Solving time grows with P.

### Command line

The notebooks call the same command-line workflow, which can also be used directly with a settings file:

```bash
uv run childcare-access check   --config config/<city>.toml    # checks files, columns, CRS, settings and unfilled values
uv run childcare-access build   --config config/<city>.toml    # inputs -> workspace/<city>/inputs (offline, no solver)
uv run childcare-access solve   --config config/<city>.toml --p 3   # the three models -> workspace/<city>/runs/P3
uv run childcare-access compare --config config/<city>.toml --run-name <run> \
    --reference-run <folder> --reference-inputs <folder>        # optional: compare two runs
```

`config/city_template.toml` is the commented settings template (every value to fill in is marked `<-- REPLACE`).
Results are written only to `workspace/`, and nothing is ever overwritten. The demo city also runs this way:
`uv run python examples/synthetic_city/make_city.py`, then the commands above with `--city synthetic_city`
(`examples/synthetic_city/README.md`).

### Your Gurobi licence

The notebooks explain how a licence is handled before they ask for one: keys are hidden while you type and are never
printed or saved in the notebook or the results; a web licence (WLS) exists as a file only during the solve; only a
salted fingerprint is recorded. Use your own computer. If a solve is interrupted, delete any leftover
`gurobi_session_...` folder in your temporary folder.

## The Münster example

`muenster_example.ipynb` shows the thesis analysis for Münster (P = 3) with all outputs saved, so it can be read
directly on GitHub: demand, vulnerability, current access, the three models' sites, efficiency and equity
indicators, walking-time distributions and maps of the savings. `docs/MUENSTER_RESULTS.md` summarises the results and
their limitations.

The Münster data are not part of this repository; `docs/MUENSTER_DATA_SOURCES.md` lists every source with its
licence and checksum. To re-run the notebook, place the model inputs in `data/muenster/model_inputs/` and the saved
solutions and results table in `results/muenster/` (the folders named in `config/muenster.toml`); these are available
from the author.

## Repository layout

| Path | Content |
| --- | --- |
| `custom_city_analysis.ipynb`, `quickstart.ipynb` | run the analysis for your city or the demo city |
| `muenster_example.ipynb` | the Münster case study (saved outputs) |
| `src/childcare_access/` | the package: data preparation (`demand`, `vulnerability`, `pedestrian_network`, `network`), the models (`all_demand_candidates`: Utilitarian and Individual Rawlsian; `district_demand_maximin`: District Rawlsian), evaluation and reporting, the workflow (`workflow`, `cli`) and the notebook helpers (`city_steps`, `city_wizard`, `guided_run`) |
| `config/` | `city_template.toml` (template), `synthetic_city.toml` (demo city), `muenster.toml` (Münster); `city_template_schema1.toml` is the older settings layout, still used by the demo and Münster files |
| `examples/synthetic_city/` | generator of the demo city and an exhaustive-enumeration check of its solutions |
| `docs/` | the method, how to run your own city, the Münster results and data sources |
| `tests/` | unit tests and the full build → solve → compare workflow on the demo city |

## Tests

```bash
uv run pytest -q tests        # about 10 minutes; tests that solve models are skipped without a Gurobi licence
```

## Limitations

- Demand is estimated potential demand (projected district totals spread over census and building data), not
  observed residences, enrolment or waiting lists.
- Access is the walking time to the nearest open facility; capacities, opening hours and parental choice are not
  modelled.
- Candidate sites are a proxy (residential locations or a supplied list), not verified developable sites.
- The priority group is a district classification; differences within a district are not represented.

## Licence

The code is licensed under the MIT licence (`LICENSE`). Data keep their own licences (`NOTICE.md`).
