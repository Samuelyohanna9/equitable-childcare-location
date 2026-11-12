"""Question-by-question creation of a city configuration for the notebook ``quickstart.ipynb``.

The answers fill the version-2 template (``config/city_template.toml``). Column names are picked from the files
themselves, and tables in common layouts are converted with ``adapters``, which only reshape and rename. No value is
estimated, and the tool never chooses a method for the user. Data are downloaded only when you choose that option.
After each file you can look at it, as a table or on a map, before answering questions about it.
"""

from __future__ import annotations

import io
import json
import re
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from . import adapters, osm_download
from .config import PROJECT_ROOT, check_city

TEMPLATE = PROJECT_ROOT / "config" / "city_template.toml"
# The German Zensus 2022 grid of children by age (one file for all of Germany), and its layout.
ZENSUS_URL = "https://www.destatis.de/static/DE/zensus/gitterdaten/Alter_in_infrastrukturellen_Altersgruppen.zip"
ZENSUS_FILE = PROJECT_ROOT / "data" / "zensus2022" / "Alter_in_infrastrukturellen_Altersgruppen.zip"
ZENSUS = {
    "member": "Zensus2022_Alter_INFR_100m-Gitter.csv",
    "id_column": "GITTER_ID_100M",
    "x_column": "x_mp_100m",
    "y_column": "y_mp_100m",
    "count_columns": ["Unter3", "a3bis5"],
    "crs": 3035,
    "cell_size_m": 100.0,
    "reference_year": 2022,
}
ZENSUS_CREDIT = (
    "Source: Statistische Ämter des Bundes und der Länder, Zensus 2022 (reference date 15 May 2022), "
    "Datenlizenz Deutschland - Namensnennung - 2.0."
)
GRID_MARGIN_M = 1000  # grid cells this far around the districts are kept
LIST_LIMIT = 40  # longer lists of values show the most common ones, and can be searched
INDICATORS = {
    "pct_welfare_15_64": "welfare recipients aged 15-64 (%)",
    "pct_welfare_0_14": "welfare recipients aged 0-14 (%)",
    "pct_children_migration_background": "children with a migration background (%)",
    "pct_children_single_parent": "children in single-parent households (%)",
}


class WizardStopped(RuntimeError):
    """The user typed q, or a needed file does not exist yet."""


class GoBack(Exception):
    """The user typed b: ask the previous question again."""


class Questions:
    """Asks until the answer is valid. ``ask`` replaces ``input`` in tests.

    ``replay`` holds answers given before (when the user went back): they are used, and shown, instead of asking,
    until they run out. ``log`` keeps every accepted answer, so going back can replay all but the last one.
    """

    def __init__(
        self, ask: Callable[[str], str] | None = None, replay: list[str] | None = None, preview: bool = False
    ) -> None:
        # The lambda looks input up at each call: Jupyter replaces input while a cell runs.
        self.ask = ask or (lambda prompt: input(prompt))  # noqa: PLW0108
        self.replay = list(replay or [])
        self.log: list[str] = []
        self._replayed = False
        self.preview = preview  # offer to look at each file after its path is given
        self.outline: gpd.GeoDataFrame | None = None  # the districts, drawn under every map once known

    def _reject(self) -> None:
        """Forget the last answer (it was not valid); stop replaying if it came from an earlier pass."""
        self.log.pop()
        if self._replayed:
            self.replay.clear()

    def text(self, question: str, default: str | None = None) -> str:
        """Ask for text; Enter takes the default. ``b`` goes back one question, ``q`` stops."""
        while True:
            hint = f" [press Enter for {default}]" if default not in (None, "") else ""
            prompt = f"{question}{hint}: "
            self._replayed = bool(self.replay)
            if self._replayed:
                answer = self.replay.pop(0)
                print(f"{prompt}{answer}")
            else:
                answer = self.ask(prompt)
            answer = answer.strip().strip('"')
            if answer.lower() == "q":
                raise WizardStopped("Stopped. The settings were not saved. Run this cell again to start over.")
            if answer.lower() == "b":
                raise GoBack
            if answer or default is not None:
                self.log.append(answer or str(default))
                return self.log[-1]
            print("  An answer is needed here (type b to go back one question, or q to stop).")

    def number(
        self,
        question: str,
        default: float | None = None,
        kind: type[float] | type[int] = float,
        low: float | None = None,
        high: float | None = None,
    ) -> float:
        """Ask for a number strictly between ``low`` and ``high``; ask again until it is one."""
        while True:
            try:
                value = kind(self.text(question, default))
            except ValueError:
                self._reject()
                print("  Please type a number.")
                continue
            if (low is None or value > low) and (high is None or value < high):
                return value
            self._reject()
            print(f"  Please type a number between {low} and {high}.")

    def choice(self, question: str, options: dict[str, str]) -> str:
        """Ask the reader to pick one of the options; return its key."""
        keys = list(options)
        print(question)
        for i, text in enumerate(options.values(), 1):
            print(f"  {i}  {text}")
        while True:
            answer = self.text(f"Type 1-{len(keys)}")
            if answer.isdigit() and 1 <= int(answer) <= len(keys):
                return keys[int(answer) - 1]
            self._reject()
            print(f"  Please type a number from 1 to {len(keys)}.")

    def column(
        self, question: str, columns: list[str], optional: bool = False, examples: dict[str, str] | None = None
    ) -> str:
        """Ask for one of the columns by number (or none, if optional); ``examples`` shows values of each."""
        print(question)
        width = max(map(len, columns), default=0)
        for i, name in enumerate(columns, 1):
            sample = (examples or {}).get(name, "")
            print(f"  {i:>2}  {name:<{width}}  {'e.g. ' + sample if sample else ''}".rstrip())
        while True:
            answer = self.text(
                "Type its number" + (", or press Enter if there is none" if optional else ""),
                "" if optional else None,
            )
            if optional and not answer:
                return ""
            if answer.isdigit() and 1 <= int(answer) <= len(columns):
                return columns[int(answer) - 1]
            if answer in columns:
                return answer
            self._reject()
            print("  That is not one of the columns listed.")

    def file(self, question: str, look: bool = True) -> Path:
        """Ask for the path of an existing file; then offer to look at it (if ``look`` and previews are on)."""
        while True:
            path = Path(self.text(question)).expanduser()
            path = path if path.is_absolute() else PROJECT_ROOT / path
            if path.is_file():
                if look and self.preview and not self._replayed:
                    self._look(path)
                return path
            self._reject()
            print(f"  No file found at {path}. Check the path, or type q to stop and prepare it first.")

    def _look(self, path: Path) -> None:
        """Offer to show the file as a table or on a map, as often as wanted, until the user goes on.

        These answers are not kept: going back from here asks for the file again.
        """
        options = ["No, go on", "Show the first rows as a table"]
        if _file_kind(path) == "map":
            options.append("Show it on a map")
        while True:
            print("Look at this file first?")
            for i, option in enumerate(options, 1):
                print(f"  {i}  {option}")
            answer = self.ask(f"Type 1-{len(options)} [press Enter to go on]: ").strip().lower()
            if answer == "q":
                raise WizardStopped("Stopped. The settings were not saved. Run this cell again to start over.")
            if answer == "b":
                raise GoBack
            if answer in ("", "1"):
                return
            if answer in ("2", "3")[: len(options) - 1]:
                try:
                    show_file(path, "map" if answer == "3" else "table", self.outline)
                except Exception as failed:  # a file that cannot be shown can still be used
                    print(f"  This file could not be shown ({failed}). You can still go on.")
            else:
                print(f"  Please type a number from 1 to {len(options)}, or press Enter to go on.")

    def values(
        self,
        question: str,
        values: list[str],
        optional: bool = False,
        counts: dict[str, int] | None = None,
        unit: str = "rows",
    ) -> list[str]:
        """Ask for some of the values by number: ``2,5`` or ``3-6``, or both (``1,3-6``).

        ``counts`` says how often each value occurs; the list then shows the most common values first. A list longer
        than ``LIST_LIMIT`` shows only those. Typing a word lists every value that contains it, and
        ``all with <word>`` takes all of them.
        """
        print(question)
        order = list(range(len(values)))
        if counts is not None:
            order.sort(key=lambda i: -counts.get(values[i], 0))
            print("  (most common first)")
        _list(values, order, counts, unit)
        if len(values) > 1:
            print("  Type a word to see only the values that contain it (for example wohn).")
        while True:
            answer = self.text(
                "Type the numbers (for example 2,5 or 3-6)" + (", or press Enter for none" if optional else ""),
                "" if optional else None,
            )
            if optional and not answer:
                return []
            picked = _pick(answer, values)
            if picked:
                return picked
            self._reject()
            word = answer[len("all with ") :] if answer.lower().startswith("all with ") else answer
            if picked is not None and word == answer:
                print(f"  Please use numbers from 1 to {len(values)}.")
                continue
            found = [i for i in order if word.lower() in values[i].lower()]
            if not found:
                print(f"  No value contains '{word}'. Type numbers from the list, or another word.")
                continue
            print(f"  {len(found):,} value(s) contain '{word}':")
            _list(values, found, counts, unit)
            print(f"  Type their numbers, or all with {word} to take all {len(found):,}.")

    def confirm(self, question: str) -> None:
        """Wait for y (go on); b goes back to the previous question."""
        while True:
            if self.text(f"{question} Type y to go on, or b to go back").lower() in ("y", "yes"):
                return
            self._reject()
            print("  Please type y, or b to go back.")


def _list(values: list[str], order: list[int], counts: dict[str, int] | None, unit: str) -> None:
    """Print the values at the positions ``order`` (at most ``LIST_LIMIT``) with their numbers in the full list."""
    width = len(str(len(values)))
    for i in order[:LIST_LIMIT]:
        value = values[i] if len(values[i]) <= 70 else values[i][:69] + "…"
        often = f"   ({counts.get(values[i], 0):,} {unit})" if counts is not None else ""
        print(f"  {i + 1:>{width}}  {value}{often}")
    if len(order) > LIST_LIMIT:
        print(f"  ... and {len(order) - LIST_LIMIT:,} more")


def _pick(answer: str, values: list[str]) -> list[str] | None:
    """The values an answer picks: numbers and ranges (``1,3-6``) or ``all with <word>``.

    Returns ``None`` if the answer is neither (a word to search for), and an empty list if it picks nothing valid.
    """
    if answer.lower().startswith("all with "):
        word = answer[len("all with ") :].strip().lower()
        return [v for v in values if word and word in v.lower()]
    picked: list[int] = []
    for part in (p.strip() for p in answer.split(",")):
        low, _, high = part.partition("-")
        if not (low.strip().isdigit() and (not high or high.strip().isdigit())):
            return None
        first, last = int(low), int(high or low)
        if not 1 <= first <= last <= len(values):
            return []
        picked += [i for i in range(first, last + 1) if i not in picked]
    return [values[i - 1] for i in picked]


def _file_kind(path: Path) -> str:
    """'table' for a CSV, 'zip' for a zip file, 'map' for anything else."""
    suffix = path.suffix.lower()
    return "table" if suffix in (".csv", ".txt") else "zip" if suffix == ".zip" else "map"


def _display(frame: pd.DataFrame) -> None:
    """Show a table: as a notebook table in Jupyter, as text elsewhere."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        get_ipython = None
    with pd.option_context("display.max_columns", 50, "display.max_colwidth", 40, "display.width", 160):
        if get_ipython is not None and get_ipython() is not None:
            display(frame)
        else:
            print(frame.to_string())


def show_file(path: Path, how: str = "table", outline: gpd.GeoDataFrame | None = None, rows: int = 10) -> None:
    """Show a data file in the notebook: its first ``rows`` rows as a table, or (``how="map"``) its shapes on a map.

    ``outline`` (the districts) is drawn under the map. Works for CSV tables, zip files with a CSV inside, and map
    files (GeoPackage, shapefile, GeoJSON and the like).
    """
    kind = _file_kind(path)
    if how == "map":
        import matplotlib.pyplot as plt

        frame = gpd.read_file(path)
        if frame.crs is None:
            raise WizardStopped(f"{path.name} has no coordinate system, so it cannot be placed on a map")
        print(f"Drawing {len(frame):,} shapes from {path.name} (a large file can take a minute) ...")
        _, ax = plt.subplots(figsize=(7, 7))
        if outline is not None:
            frame = frame.to_crs(outline.crs)
            outline.boundary.plot(ax=ax, color="0.45", linewidth=0.8)
        frame.plot(ax=ax, color="tab:blue", markersize=4, linewidth=0.4)
        ax.set_title(path.name + ("  (grey: your districts)" if outline is not None else ""))
        ax.set_axis_off()
        plt.show()
        return
    if kind == "table":
        frame = _read_table(path, rows)[0]
        print(f"The first {len(frame)} rows of {path.name}:")
    elif kind == "zip":
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise WizardStopped(f"{path.name} contains no CSV file")
            with archive.open(members[0]) as raw:
                handle = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace")
                frame = pd.read_csv(handle, sep=None, engine="python", nrows=rows, dtype=str)
        print(f"The first {len(frame)} rows of {members[0]} (in {path.name}; {len(members)} CSV file(s) inside):")
    else:
        info = pyogrio.read_info(path)
        frame = pyogrio.read_dataframe(path, read_geometry=False, max_features=rows)
        print(f"The first {len(frame)} of {info['features']:,} rows of {path.name} (the shapes are not shown here):")
    _display(frame)


def _examples(values: pd.Series) -> str:
    """Up to three different values of a column, shortened, to help pick the right one."""
    shown = []
    for value in values.dropna().astype(str).str.strip():
        if value and value not in shown:
            shown.append(value if len(value) <= 24 else value[:23] + "…")
        if len(shown) == 3:
            break
    return ", ".join(shown)


def _map_columns(path: Path, folder: Path, stem: str) -> tuple[Path, list[str], dict[str, str]]:
    """The map file to use (a copy if it repeats a column name), its column names and example values."""
    used, columns = adapters.map_with_unique_columns(path, folder, stem)
    if used != path:
        print(
            f"  This file has two or more columns with the same name. Here they are numbered ({', '.join(columns)}),\n"
            f"  and a copy with these names is used: {_path(used)}"
        )
    sample = pyogrio.read_dataframe(used, read_geometry=False, max_features=20)
    return used, columns, {c: _examples(sample[c]) for c in columns if c in sample}


def _table_examples(frame: pd.DataFrame) -> dict[str, str]:
    """Example values of every column of a table."""
    return {str(c): _examples(frame[c]) for c in frame.columns}


def _path(path: Path) -> str:
    """A path as written into the settings: relative to the project root when it lies inside it."""
    with_root = path.resolve()
    return (
        with_root.relative_to(PROJECT_ROOT).as_posix()
        if with_root.is_relative_to(PROJECT_ROOT)
        else with_root.as_posix()
    )


def _read_table(path: Path, nrows: int | None = None) -> tuple[pd.DataFrame, str, str]:
    """A CSV table with its separator and encoding detected."""
    for encoding in ("utf-8", "cp1252"):
        try:
            frame = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
                nrows=nrows,
                dtype=str if nrows == 0 else None,
            )
            with path.open(encoding=encoding) as handle:
                head = handle.readline()
            return frame, max(",;\t|", key=head.count), encoding
        except UnicodeDecodeError:
            continue
    raise WizardStopped(f"{path.name} could not be read as text. Save it as a CSV file and try again.")


def _decimal(path: Path, separator: str, encoding: str) -> str:
    """The decimal mark of a table: "," if numbers like 6,4 occur in it (possible only with another separator)."""
    if separator == ",":
        return "."
    sample = path.read_text(encoding=encoding, errors="replace")[:200_000]
    return (
        "," if re.search(rf"(^|{re.escape(separator)})\s*-?\d+,\d+\s*({re.escape(separator)}|$)", sample, re.M) else "."
    )


def to_toml(data: dict) -> str:
    """Nested tables of strings, numbers, booleans and lists as TOML."""

    def value(v: object) -> str:
        """One value in TOML syntax."""
        return ("true" if v else "false") if isinstance(v, bool) else json.dumps(v, ensure_ascii=False)

    sections = []

    def walk(table: dict, prefix: str) -> None:
        """Write one table (and its subtables) as TOML lines."""
        lines = [f"{k} = {value(v)}" for k, v in table.items() if not isinstance(v, dict)]
        if prefix:
            sections.append(f"[{prefix}]\n" + "\n".join(lines))
        for k, v in table.items():
            if isinstance(v, dict):
                walk(
                    v,
                    f"{prefix}.{k}" if prefix else (k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)),
                )

    walk(data, "")
    return "# Created by the question-by-question setup of quickstart.ipynb.\n\n" + "\n\n".join(sections) + "\n"


def _network_help(districts: gpd.GeoDataFrame) -> str:
    """How to get a walking network for these districts from Overpass Turbo (with the query)."""
    west, south, east, north = districts.to_crs(4326).total_bounds
    return (
        "You need a walking network file. To get one:\n"
        "  1. Open https://overpass-turbo.eu\n  2. Paste this query and click Run:\n\n"
        "     "
        f"[out:json][timeout:600][bbox:{south - 0.01:.4f},{west - 0.01:.4f},{north + 0.01:.4f},{east + 0.01:.4f}];\n"
        f'     (way["highway"]["highway"!~"^({osm_download.EXCLUDED_HIGHWAYS})$"];>;);\n     out body;\n\n'
        "  3. Choose Export, then 'raw data', and save the file.\n"
        "For a large city this often fails; then choose 'download it for me', which fetches the city in parts.\n"
        "OpenStreetMap data are under the Open Database License: credit OpenStreetMap contributors.\n"
        "Run this cell again when you have the file."
    )


def _districts(q: Questions, config: dict, folder: Path) -> gpd.GeoDataFrame:
    """Questions on the district file; returns the districts."""
    print("\n--- Districts: a map file with one shape per district (for example .gpkg, .shp or .geojson)")
    path = q.file("Path of the district file")
    info = pyogrio.read_info(path)
    if info["crs"] is None:
        raise WizardStopped(f"{path.name} has no coordinate system. Set one in GIS software, then try again.")
    path, columns, examples = _map_columns(path, folder, "districts")
    table = config["sources"]["districts"]
    table.update(
        file=_path(path),
        id_column=q.column("Which column holds the district ID?", columns, examples=examples),
        name_column=q.column("Which column holds the district name?", columns, examples=examples),
        id_width=0,
    )
    districts = gpd.read_file(path)
    q.outline = districts
    if districts[table["id_column"]].duplicated().any():
        raise WizardStopped("District IDs must be unique, and that column has repeated values.")
    suggested = districts.estimate_utm_crs()
    config["crs"]["projected"] = q.number(
        "Projected coordinate system (EPSG code, in metres)", suggested.to_epsg(), int
    )
    print(f"Found {len(districts)} districts.")
    q.confirm(
        f"IDs look like {_examples(districts[table['id_column']])}; names look like "
        f"{_examples(districts[table['name_column']])}."
    )
    return districts


def _download(districts: gpd.GeoDataFrame, target: Path) -> tuple[Path, str]:
    """Download the walking network into ``target`` (or reuse it if it is there); returns the file and its date."""
    if target.is_file():
        stamp = json.loads(target.read_text(encoding="utf-8")).get("osm3s", {}).get("timestamp_osm_base", "")
        if stamp:
            print(f"Using the walking network downloaded before: {_path(target)} (map data of {stamp}).")
            return target, stamp
    try:
        figures = osm_download.download_network(districts, target)
    except osm_download.DownloadFailed as failed:
        raise WizardStopped(
            f"The download did not finish: {failed}\nRun this cell again later: the parts already downloaded are "
            "kept, so it continues where it stopped.\n\nOr get the file yourself:\n" + _network_help(districts)
        ) from None
    return target, figures["date"]


def _network(q: Questions, config: dict, districts: gpd.GeoDataFrame, folder: Path) -> None:
    """Questions on the walking network."""
    print("\n--- Walking network: OpenStreetMap paths as an Overpass JSON file")
    have = q.choice(
        "Do you have this file?",
        {
            "yes": "Yes",
            "download": "No, download it for me now (needs internet; a few minutes, longer for a large city)",
            "help": "No, show me how to get it myself",
        },
    )
    if have == "help":
        raise WizardStopped(_network_help(districts))
    if have == "download":
        path, stamp = _download(districts, folder.parent / "osm" / "highways.json")
        config["osm"].update(network_snapshot=_path(path), snapshot_timestamp=stamp)
        return
    path = q.file("Path of the network file", look=False)
    stamp = ""
    try:
        stamp = json.loads(path.read_text(encoding="utf-8")).get("osm3s", {}).get("timestamp_osm_base", "")
    except (ValueError, UnicodeDecodeError):
        raise WizardStopped(f"{path.name} is not a JSON file. Export the Overpass result as raw data.") from None
    config["osm"].update(
        network_snapshot=_path(path),
        snapshot_timestamp=q.text("Date of this OpenStreetMap data", stamp or None),
    )


def _facilities(q: Questions, config: dict, folder: Path) -> None:
    """Questions on the existing facilities."""
    print("\n--- Existing childcare facilities: a map file of points or shapes")
    path, fields, examples = _map_columns(q.file("Path of the facilities file"), folder, "facilities")
    facilities = pyogrio.read_dataframe(path, read_geometry=False)
    while True:
        id_column = q.column(
            "Which column holds a unique facility ID? (press Enter if there is none: they are then numbered)",
            fields,
            optional=True,
            examples=examples,
        )
        if not id_column or not facilities[id_column].duplicated().any():
            break
        q._reject()
        print(
            f"  {id_column} is not unique: {facilities[id_column].nunique():,} different values for "
            f"{len(facilities):,} facilities. Pick a column with one value per facility, or press Enter."
        )
    config["facilities"].update(source="file", file=_path(path), id_column=id_column or "facility_id")


def _labels(values: pd.Series) -> set[str]:
    """District IDs as clean text (spaces stripped, a trailing ``.0`` from number columns removed)."""
    return set(values.dropna().astype(str).str.strip().str.replace(r"\.0$", "", regex=True))


def _id_problem(given: set[str], known: set[str], every: bool) -> str:
    """What differs between the IDs of a table and the district IDs (empty if nothing does)."""
    extra, missing = sorted(given - known), sorted(known - given) if every else []
    lines = []
    if extra:
        lines.append(f"{len(extra):,} ID(s) of this table are not district IDs, e.g. {', '.join(extra[:5])}")
    if missing:
        lines.append(f"{len(missing):,} district(s) have no row in this table, e.g. {', '.join(missing[:5])}")
    if lines and {i.lstrip("0") for i in given} == {i.lstrip("0") for i in known}:
        lines.append("The IDs differ only in leading zeros (01 and 1): write them the same way in both files.")
    return "\n    ".join(lines)


def _district_id_column(
    q: Questions,
    config: dict,
    districts: gpd.GeoDataFrame,
    path: Path,
    frame: pd.DataFrame,
    every: bool = True,
    first_word: bool = False,
) -> str:
    """Ask which column of a district table holds the district ID, and check it against the district file at once.

    ``every``: each district needs a row (not so for a list of priority districts). ``first_word``: only the first
    word of each value is the ID (for example "01 Altstadt").
    """
    columns, examples = [str(c) for c in frame.columns], _table_examples(frame)
    _, separator, encoding = _read_table(path, 0)
    full = pd.read_csv(path, sep=separator, encoding=encoding, dtype=str)
    full.columns = columns
    column = q.column("Which column holds the district ID?", columns, examples=examples)
    while True:
        values = full[column].str.extract(r"^\s*(\S+)", expand=False) if first_word else full[column]
        id_column = config["sources"]["districts"]["id_column"]
        given, known = _labels(values), _labels(districts[id_column])
        problem = _id_problem(given, known, every)
        if not problem:
            return column
        print(
            "  These IDs do not match the district IDs. The tool joins the tables by these IDs, so they must be the "
            f"same.\n    This table, column {column}: e.g. {_examples(values)}\n"
            f"    District file, column {id_column}: e.g. {_examples(districts[id_column])}\n    {problem}"
        )
        fix = q.choice(
            "What do you want to do?",
            {
                "table": "Pick another column of this table",
                "districts": "Pick another ID column of the district file",
                "keep": "Keep these (I will correct a file myself before building)",
            },
        )
        if fix == "keep":
            return column
        if fix == "table":
            column = q.column("Which column holds the district ID?", columns, examples=examples)
            continue
        fields = [c for c in districts.columns if c != districts.geometry.name]
        chosen = q.column(
            "Which column of the district file holds the district ID?",
            fields,
            examples=_table_examples(districts[fields].head(50)),
        )
        if districts[chosen].duplicated().any():
            print("  That column has repeated values, and district IDs must be unique.")
            continue
        config["sources"]["districts"]["id_column"] = chosen


def _population(q: Questions, config: dict, districts: gpd.GeoDataFrame) -> None:
    """Questions on the children per district."""
    print("\n--- Children per district: a CSV table with one row per district")
    path = q.file("Path of the table")
    frame, separator, encoding = _read_table(path, 20)
    columns, examples = [str(c) for c in frame.columns], _table_examples(frame)
    table = config["population"]
    table.update(
        file=_path(path),
        layout="wide",
        separator=separator,
        encoding=encoding,
        district_id_column=_district_id_column(q, config, districts, path, frame),
    )
    split = q.choice(
        "How are the children counted?",
        {"one": "One column for ages 0-5", "two": "Two columns, ages 0-2 and 3-5"},
    )
    if split == "one":
        table.update(
            children_0_5=q.column("Which column holds children aged 0-5?", columns, examples=examples),
            age_0_2="",
            age_3_5="",
        )
    else:
        table.update(
            children_0_5="",
            age_0_2=q.column("Which column holds children aged 0-2?", columns, examples=examples),
            age_3_5=q.column("Which column holds children aged 3-5?", columns, examples=examples),
        )
    table.update(
        count_type=q.choice(
            "Are these counts observed or projected?",
            {"observed": "Observed", "projected": "Projected"},
        ),
        reference_year=q.number("Which year do they refer to?", kind=int),
        source=q.text("Where do they come from (publisher, table, date)?"),
    )


def download_zensus(target: Path = ZENSUS_FILE, url: str = ZENSUS_URL) -> Path:
    """Download the German Zensus 2022 grid once (about 38 MB); a complete earlier download is used again."""
    if target.is_file() and zipfile.is_zipfile(target):
        print(f"Using the Zensus 2022 grid downloaded before: {_path(target)}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    print("Downloading the Zensus 2022 grid for Germany (about 38 MB) ...")
    request = urllib.request.Request(url, headers={"User-Agent": "childcare-access"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            done, step = 0, 5_000_000
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if done // step != (done - len(chunk)) // step:
                    print(f"  {done / 1e6:.0f} MB")
    except OSError as failed:
        partial.unlink(missing_ok=True)
        raise WizardStopped(
            f"The download did not work ({failed}). Check the internet connection and run this cell again, or download "
            f"the file yourself from {url} and give its path."
        ) from None
    if not zipfile.is_zipfile(partial):
        partial.unlink()
        raise WizardStopped(f"The file from {url} is not a zip file. Download it yourself and give its path.")
    partial.replace(target)
    print(f"Saved {_path(target)}. {ZENSUS_CREDIT}")
    return target


def keep_cells_near(
    source: Path,
    member: str,
    districts: gpd.GeoDataFrame,
    columns: dict,
    target: Path,
    margin_m: float = GRID_MARGIN_M,
) -> int:
    """Keep the grid cells whose centre lies within ``margin_m`` of the districts' bounding box.

    ``columns`` holds ``x_column``, ``y_column`` and ``crs`` of the grid. The kept rows are written unchanged, as one
    CSV (``census_grid.csv``, separator ``;``) in the zip ``target``; the number of rows is returned. The result is
    reused when the same source and area were prepared before (going back in the questions does not redo it).
    """
    box = districts.to_crs(int(columns["crs"])).total_bounds + np.array([-margin_m, -margin_m, margin_m, margin_m])
    stamp = {"source": str(source), "member": member, "box": [round(float(v), 3) for v in box]}
    record = target.with_suffix(".json")
    if target.is_file() and record.is_file():
        earlier = json.loads(record.read_text(encoding="utf-8"))
        if {k: earlier.get(k) for k in stamp} == stamp:
            return int(earlier["cells"])
    print("Keeping the grid cells around your city (this reads the whole grid once; up to a minute) ...")
    with zipfile.ZipFile(source) as archive, archive.open(member) as raw:
        handle = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace")
        separator = max(",;\t|", key=handle.readline().count)
    parts = []
    with zipfile.ZipFile(source) as archive, archive.open(member) as raw:
        handle = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace")
        for chunk in pd.read_csv(handle, sep=separator, chunksize=1_000_000, dtype=str):
            x = pd.to_numeric(chunk[columns["x_column"]], errors="coerce")
            y = pd.to_numeric(chunk[columns["y_column"]], errors="coerce")
            parts.append(chunk[(x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])])
    cells = pd.concat(parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("census_grid.csv", cells.to_csv(sep=";", index=False))
    record.write_text(json.dumps({**stamp, "cells": len(cells)}, indent=1), encoding="utf-8")
    return len(cells)


def _zensus(config: dict, districts: gpd.GeoDataFrame, folder: Path) -> None:
    """Settings for the downloaded Zensus 2022 grid, cut to the cells around the city."""
    source = download_zensus()
    target = folder / "census_grid.zip"
    with zipfile.ZipFile(source) as archive:
        member = next((m for m in archive.namelist() if m.endswith(ZENSUS["member"])), None)
    if member is None:
        raise WizardStopped(f"{_path(source)} does not contain {ZENSUS['member']}. Delete it and run this cell again.")
    count = keep_cells_near(source, member, districts, ZENSUS, target)
    if not count:
        raise WizardStopped(
            "No Zensus grid cells lie near your districts: the Zensus grid covers Germany only.\n"
            "Run this cell again and, for where children live, choose 'District totals and building footprints'."
        )
    print(f"Kept {count:,} grid cells around your city: {_path(target)}\n{ZENSUS_CREDIT}")
    config["demand"]["census"].update(
        file=_path(target),
        sha256="",
        member_suffix="census_grid.csv",
        separator=";",
        encoding="utf-8",
        **{k: v for k, v in ZENSUS.items() if k != "member"},
    )


def _census(q: Questions, config: dict, districts: gpd.GeoDataFrame, folder: Path) -> None:
    """Questions on the census grid."""
    print("\n--- Census grid: children per grid cell, with cell-centre coordinates (a CSV inside a zip file)")
    have = q.choice(
        "Do you have this file?",
        {
            "yes": "Yes, I will give the path of the zip file",
            "zensus": "No, download the German Zensus 2022 grid for me (Germany only; about 38 MB, used for any German "
            "city)",
        },
    )
    if have == "zensus":
        _zensus(config, districts, folder)
        return
    path = q.file("Path of the zip file")
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise WizardStopped(f"{path.name} contains no CSV file.")
        member = members[0] if len(members) == 1 else q.values("Which CSV file is the grid? Pick one.", members)[0]
        with archive.open(member) as handle:
            head = handle.readline().decode("utf-8-sig", "replace").strip()
            rows = [handle.readline().decode("utf-8", "replace").strip() for _ in range(3)]
    separator = max(",;\t|", key=head.count)
    columns = [c.strip('"') for c in head.split(separator)]
    cells = [[v.strip('"') for v in row.split(separator)] for row in rows if row]
    examples = {c: ", ".join(r[i] for r in cells if i < len(r)) for i, c in enumerate(columns)}
    table = config["demand"]["census"]
    table.update(
        file=_path(path),
        member_suffix=Path(member).name,
        separator=separator,
        encoding="utf-8-sig",
        id_column=q.column("Which column holds the cell ID?", columns, examples=examples),
        x_column=q.column("Which column holds the cell-centre x coordinate?", columns, examples=examples),
        y_column=q.column("Which column holds the cell-centre y coordinate?", columns, examples=examples),
        count_columns=q.values("Which columns hold the children? Together they must cover ages 0-5.", columns),
        crs=q.number("EPSG code of these coordinates", 3035, int),
        cell_size_m=q.number("Cell size in metres", 100, float, 0),
        reference_year=q.number("Census year", kind=int),
    )


def _direct(q: Questions, config: dict, folder: Path) -> None:
    """Questions on directly supplied child locations."""
    print("\n--- Child locations: a map file of points, each with a number of children")
    path = q.file("Path of the file")
    info = pyogrio.read_info(path)
    if info["crs"] is None:
        raise WizardStopped(f"{path.name} has no coordinate system. Set one in GIS software, then try again.")
    path, columns, examples = _map_columns(path, folder, "child_locations")
    table = config["demand"]["direct"]
    table.update(
        geometry="points",
        file=_path(path),
        id_column=q.column("Which column holds the location ID?", columns, examples=examples),
        children_0_5=q.column("Which column holds children aged 0-5?", columns, examples=examples),
        age_0_2="",
        age_3_5="",
        district_column="",
        count_type=q.choice(
            "Are these counts observed or projected?",
            {"observed": "Observed", "projected": "Projected"},
        ),
        reference_year=q.number("Which year do they refer to?", kind=int),
        source=q.text("Where do they come from (register, date)?"),
    )


def _buildings(q: Questions, config: dict, folder: Path, purpose: str = "") -> None:
    """Questions on the building file; ``purpose`` says what the buildings are used for."""
    print("\n--- Buildings: a map file of building footprints (shapes, not points)")
    if purpose:
        print(f"The residential buildings are used {purpose}.")
    path = q.file("Path of the buildings file")
    layers = [name for name, _ in pyogrio.list_layers(path)]
    layer = layers[0] if len(layers) == 1 else q.values("Which layer holds the buildings? Pick one.", layers)[0]
    footprints = gpd.read_file(path, layer=layer)
    geometry = footprints.geometry.name
    footprints.columns = [
        c if c == geometry else n
        for c, n in zip(footprints.columns, adapters.unique_names(footprints.columns), strict=True)
    ]
    columns = [c for c in footprints.columns if c != geometry]
    examples = _table_examples(footprints[columns].head(50))
    id_column = q.column("Which column holds a unique building ID?", columns, optional=True, examples=examples)
    if (
        q.choice(
            "Are all these buildings residential?",
            {"yes": "Yes", "no": "No, a column says what each building is used for"},
        )
        == "yes"
    ):
        residential, mixed = pd.Series(True, index=footprints.index), False
    else:
        use = q.column("Which column says what each building is used for?", columns, examples=examples)
        uses = footprints[use].dropna().astype(str)
        kinds, counts = sorted(uses.unique()), uses.value_counts().to_dict()
        if len(kinds) > len(footprints) / 2:
            print(
                f"  Note: this column has {len(kinds):,} different values for {len(footprints):,} rows, so it is\n"
                "  probably not a list of building uses. Building footprints usually have a short list of uses."
            )
        chosen = q.values(
            "Which values mean residential? Include mixed use with housing. Pick as many as apply.",
            kinds,
            counts=counts,
            unit="buildings",
        )
        print(f"  {int(uses.isin(chosen).sum()):,} of {len(footprints):,} buildings have these values.")
        partly = q.values(
            "Which of those are mixed use with housing?", chosen, optional=True, counts=counts, unit="buildings"
        )
        residential, mixed = (
            footprints[use].astype(str).isin(chosen),
            footprints[use].astype(str).isin(partly),
        )
    prepared = folder / "buildings.gpkg"
    adapters.buildings_layer(footprints, residential, mixed, id_column or None).to_file(
        prepared, layer="buildings", driver="GPKG"
    )
    print(f"Prepared {int(residential.sum()):,} residential buildings of {len(footprints):,}: {_path(prepared)}")
    config["buildings"].update(
        file=_path(prepared),
        layer="buildings",
        id_column="building_id",
        function_column="function",
        mixed_use_code="mixed",
        object_type_column="object_type",
        building_value="building",
        part_value="part",
        sha256="",
        residential_codes={"residential": "Residential", "mixed": "Mixed use with housing"},
    )


def _vulnerability(q: Questions, config: dict, folder: Path, districts: gpd.GeoDataFrame) -> None:
    """Questions on the priority group."""
    print("\n--- Priority group: which districts count as the most vulnerable")
    mode = q.choice(
        "Which data do you have?",
        {
            "four_indicator": "The default index: four percentages per district (the concepts are listed in README.md)",
            "external_index": "One documented index per district",
            "priority_districts": "A fixed list of priority districts",
        },
    )
    table = config["vulnerability"]
    table["mode"] = mode
    if mode == "four_indicator":
        print(
            "Each column must measure the concept the question names. A different concept changes what the index means."
        )
        path = q.file("Path of the CSV table, with one row per district")
        frame, _, _ = _read_table(path)
        columns, examples = [str(c) for c in frame.columns], _table_examples(frame)
        id_column = _district_id_column(q, config, districts, path, frame, first_word=True)
        mapping = {
            key: q.column(f"Which column holds {label}?", columns, examples=examples)
            for key, label in INDICATORS.items()
        }
        year = q.number("Which year do these values refer to?", kind=int)
        source = q.text("Where do they come from (publisher, table, definition)?")
        prepared = folder / "indicators.csv"
        adapters.indicators_long(frame, id_column, mapping, str(year), year=year).to_csv(prepared, index=False)
        table["four_indicator"].update(
            file=_path(prepared),
            year=year,
            quantile=q.number(
                "Priority share: index at or above this quantile (0.75 = the top quarter, the default)",
                0.75,
                float,
                0,
                1,
            ),
            separator=",",
            encoding="utf-8",
            decimal=".",
            area_column="area",
            time_column="date",
            indicator_column="indicator",
            value_column="value",
            district_code_pattern=r"^(\S+)",
            indicators=mapping,
            indicator_definitions=dict.fromkeys(mapping, source),
        )
    elif mode == "external_index":
        path = q.file("Path of the CSV table")
        frame, separator, encoding = _read_table(path, 20)
        columns, examples = [str(c) for c in frame.columns], _table_examples(frame)
        table["external_index"].update(
            file=_path(path),
            separator=separator,
            encoding=encoding,
            decimal=_decimal(path, separator, encoding),
            district_id_column=_district_id_column(q, config, districts, path, frame),
            value_column=q.column("Which column holds the index?", columns, examples=examples),
            name=q.text("Name and year of the index"),
            definition=q.text("Publisher and method of the index"),
            higher_is_more_vulnerable=q.choice("Does a higher value mean more vulnerable?", {"yes": "Yes", "no": "No"})
            == "yes",
            quantile=q.number(
                "Priority share: index at or above this quantile (0.75 = the top quarter)",
                0.75,
                float,
                0,
                1,
            ),
        )
    else:
        path = q.file("Path of the CSV table, with one row per priority district")
        frame, separator, encoding = _read_table(path, 20)
        columns, examples = [str(c) for c in frame.columns], _table_examples(frame)
        table["priority_districts"].update(
            file=_path(path),
            separator=separator,
            encoding=encoding,
            district_id_column=_district_id_column(q, config, districts, path, frame, every=False),
            label=q.text("Name of the programme or list"),
            definition=q.text("Source and selection rule"),
        )
    for other in ("four_indicator", "external_index", "priority_districts"):
        if other != mode:
            del table[other]


def _clear_screen() -> None:
    """Clear the cell output in a notebook (nothing happens elsewhere)."""
    try:
        from IPython import get_ipython
        from IPython.display import clear_output
    except ImportError:
        return
    if get_ipython() is not None:
        clear_output()


def create_city(ask: Callable[[str], str] | None = None, preview: bool | None = None) -> Path:
    """Ask for everything a city needs, write ``config/<city>.toml``, check it, and return its path.

    Typing b at a question goes back one question: the questions are asked again from the start, with the earlier
    answers filled in, up to the one before. ``preview`` offers to look at each file; it is on when the user types
    the answers (``ask`` not given).
    """
    given: list[str] = []
    while True:
        q = Questions(ask, replay=given, preview=ask is None if preview is None else preview)
        try:
            return _create_city(q)
        except GoBack:
            given = q.log[:-1]
            _clear_screen()
            print("Going back one question. Your earlier answers are filled in below.\n")


def _summary(config: dict) -> str:
    """The main choices of the settings, one per line, to check before saving."""
    demand, vulnerability = config["demand"]["mode"], config["vulnerability"]["mode"]
    d = config["sources"]["districts"]
    lines = [
        f"  City: {config['city']['name']}",
        f"  Districts: {d['file']} (ID column {d['id_column']}, name column {d['name_column']})",
        f"  Coordinate system: EPSG {config['crs']['projected']}",
        f"  Walking network: {config['osm']['network_snapshot']} (map data of {config['osm']['snapshot_timestamp']})",
        f"  Existing childcare: {config['facilities']['file']}",
        f"  Where children live: {demand}",
    ]
    if "population" in config:
        lines.append(f"  Children per district: {config['population']['file']}")
    lines += [
        f"  Where new facilities can go: {config['candidates']['source']}",
        f"  Priority group: {vulnerability} ({config['vulnerability'][vulnerability]['file']})",
    ]
    return "\n".join(lines)


def _create_city(q: Questions) -> Path:
    """The questions of :func:`create_city`, in order."""
    print(
        "This creates the settings file for your city, one question at a time.\n"
        "Made a mistake? Type b to go back one question. Type q to stop.\n"
        "After each file you can look at it as a table or on a map.\n"
        "You need your data files on this computer; see README.md for the list. The walking network and, in Germany,\n"
        "the census grid can be downloaded for you."
    )
    config = tomllib.loads(TEMPLATE.read_text(encoding="utf-8"))
    name = q.text("\nCity name")
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "my_city"
    path = PROJECT_ROOT / "config" / f"{slug}.toml"
    if path.exists():
        raise WizardStopped(
            f"{_path(path)} already exists. To use it, run this cell again and give its path. "
            "To change it, edit the file, or delete it and answer the questions again."
        )
    folder = PROJECT_ROOT / "data" / slug / "prepared"
    folder.mkdir(parents=True, exist_ok=True)
    config["city"].update(name=name, scenario=slug)
    config["workspace"]["root"] = f"workspace/guided/{slug}"
    districts = _districts(q, config, folder)
    _network(q, config, districts, folder)
    _facilities(q, config, folder)
    mode = q.choice(
        "\nHow do you want to estimate where children live?",
        {
            "census_hybrid": "Census grid and buildings (the default; in Germany the grid can be downloaded for you)",
            "building_area": "District totals and building footprints (footprint allocation)",
            "direct": "I have child locations already",
        },
    )
    config["demand"]["mode"] = mode
    if mode == "census_hybrid":  # the grid first: in Germany it can be downloaded right away
        _census(q, config, districts, folder)
    if mode in ("census_hybrid", "building_area"):
        _population(q, config, districts)
    else:
        del config["population"]
    if mode == "direct":
        _direct(q, config, folder)
    for method in ("census", "direct"):
        wanted = {"census": "census_hybrid", "direct": "direct"}[method]
        if mode != wanted:
            del config["demand"][method]
    print(
        "\n--- Candidate sites: the places where the model may open a new facility.\n"
        "This does not change where children live; the question on where children live decided that."
    )
    source = q.choice(
        "Where may the model open new facilities?",
        {
            "residential_buildings": "At any residential building "
            + ("(needs a buildings file)" if mode == "direct" else "(the default)"),
            "file": "Only at sites listed in a file (for example plots that are available for building)",
        },
    )
    config["candidates"]["source"] = source
    if source == "file":
        site, fields, examples = _map_columns(
            q.file("Path of the sites file (a map file of points or shapes)"), folder, "sites"
        )
        config["candidates"].update(
            file=_path(site),
            id_column=q.column("Which column holds the site ID?", fields, examples=examples),
            name_column="",
        )
    if mode != "direct" or source == "residential_buildings":
        uses = []
        if mode != "direct":
            uses.append("to spread the children of each district over its residential buildings")
        if source == "residential_buildings":
            uses.append("as the candidate sites for new facilities")
        _buildings(q, config, folder, " and ".join(uses))
    else:
        del config["buildings"]
    _vulnerability(q, config, folder, districts)
    config["model_parameters"]["facility_budget"] = 3
    config["solver_resources"]["nodefile_min_free_gb"] = 10.0
    print("\n--- Check your answers\n" + _summary(config))
    q.confirm("Save these settings?")
    path.write_text(to_toml(config), encoding="utf-8")
    print(f"\nSaved {_path(path)}. Next time you can type this path instead of answering again.")
    ok, lines = check_city(path)
    if not ok:
        print("\n".join(f"  {line[:170]}" for line in lines))
        print(
            "The settings were saved, but the check above found problems. Fix them in the file or delete it and "
            "start again."
        )
    return path
