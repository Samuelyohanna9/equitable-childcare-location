"""Section-by-section city setup for the notebook ``custom_city_analysis.ipynb``.

Each section explains one input, lets the user pick a file, shows what was found and checks it at once. Answers
are saved after every section, so a mistake in one section does not mean starting again. The settings file itself is
written by ``city_wizard.create_city``, which receives the saved answers, so both notebooks produce the same settings.
Tables are only reshaped and filtered: no value is estimated, and the tool never chooses a method for the user.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import geopandas as gpd
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from IPython.display import HTML, display

from . import adapters, osm_download
from . import city_wizard as wizard
from . import guided_run as guide
from .config import PROJECT_ROOT, check_city

MAP_FILES = (".gpkg", ".geojson", ".json", ".shp", ".fgb")
TABLES = (".csv", ".txt", ".tsv")
LARGE_CITY = 50000  # residential buildings; Dortmund (113,917) needed about 2 hours to build and a large solver model
CENSUS_MARGIN_M = 1000
TECHNICAL_COLUMNS = (
    "rowid",
    "objectid",
    "object_id",
    "fid",
    "ogc_fid",
)  # record numbers of a portal, never district IDs
DEMAND = {
    "census_hybrid": "Census grid and buildings (the default method, census-grid allocation)",
    "building_area": "District totals and buildings, for cities without a census grid (footprint allocation)",
    "direct": "I already have child locations",
}
PRIORITY = {
    "four_indicator": "The default index: four percentages per district",
    "external_index": "One documented index per district",
    "priority_districts": "A fixed list of priority districts",
}
# What the callbacks of a file section receive and return (see _file_section).
Preview = Callable[[], object]  # shows what the chosen file holds
Reader = Callable[[str], tuple[list[str], Preview]]  # file -> its columns and a preview
FieldMaker = Callable[[list[str]], dict[str, widgets.Widget]]  # columns -> the input boxes of the form
Saver = Callable[[str, dict[str, Any]], str]  # file, form values -> confirmation text
Widgets = dict[str, widgets.Widget]  # what a section returns, so tests and the notebook can drive it
STARTS = {
    "new": "My own city: start, or continue a saved setup",
    "file": "My own city: I have a finished settings file",
    "demo": "Demo city Testburg (small, synthetic, about 1 minute)",
}


class StepProblem(RuntimeError):
    """Something the reader can correct; the message says what."""


def _labels(series: pd.Series) -> pd.Series:
    """District labels as clean text (spaces stripped, a trailing ``.0`` from float columns removed)."""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _number(columns: list[str], name: str) -> str:
    """1-based position of a column, as the reader sees it in a table preview."""
    return str(columns.index(name) + 1)


def _separator(head: str) -> str:
    """The most frequent of the usual column separators in a header line."""
    return max(",;\t|", key=head.count)


def _read_unique(file: str | Path, **options: Any) -> gpd.GeoDataFrame:
    """Read a map file with repeated column names numbered (``adapters.unique_names``), so each can be chosen."""
    frame = gpd.read_file(file, **options)
    geometry = frame.geometry.name if isinstance(frame, gpd.GeoDataFrame) else None
    frame.columns = [
        c if c == geometry else n for c, n in zip(frame.columns, adapters.unique_names(frame.columns), strict=True)
    ]
    return frame


def _fields(file: str | Path, **options: Any) -> list[str]:
    """The column names of a map file, repeated names numbered as in :func:`_read_unique`."""
    return adapters.unique_names(pyogrio.read_info(file, **options)["fields"])


def _column_values(file: str | Path, layer: str, column: str) -> pd.Series:
    """One column of a map layer, without geometry (``column`` as numbered by :func:`_fields`)."""
    fields = [str(f) for f in pyogrio.read_info(file, layer=layer)["fields"]]
    if fields.count(column) == 1:
        return pyogrio.read_dataframe(file, layer=layer, columns=[column], read_geometry=False)[column]
    frame = pyogrio.read_dataframe(file, layer=layer, read_geometry=False)
    frame.columns = adapters.unique_names(frame.columns)
    return frame[column]


class Setup:
    """The answers of one city, kept in ``data/<city>/setup_answers.json``.

    No widgets here, so all of it can be tested.
    """

    def __init__(self, name: str = "") -> None:
        self.kind, self.settings_path = "", None
        self.name, self.slug, self.answers = "", "", {}
        if name:
            self.open(name)

    def open(self, name: str) -> None:
        """Start or continue the setup of the city ``name``; its answers are loaded if saved before."""
        name = name.strip()
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if not slug:
            raise StepProblem("Please type a city name.")
        self.kind, self.name, self.slug = "new", name, slug
        self.folder = PROJECT_ROOT / "data" / slug
        self.prepared = self.folder / "prepared"
        self.prepared.mkdir(parents=True, exist_ok=True)
        self.answers_file = self.folder / "setup_answers.json"
        self.answers = json.loads(self.answers_file.read_text(encoding="utf-8")) if self.answers_file.is_file() else {}

    def save(self, step: str, **values: Any) -> None:
        """Save the answers of one section (written to disk at once)."""
        self.answers[step] = values
        self.answers_file.write_text(json.dumps(self.answers, indent=1, ensure_ascii=False), encoding="utf-8")

    def need(self, *steps: str) -> None:
        """Stop with a message unless the city was chosen and the given sections are saved."""
        if not self.kind:
            raise StepProblem("Start with section 1 and choose your city.")
        if self.kind != "new":
            raise StepProblem("Not needed for this choice. Go to section 13.")
        missing = [s for s in steps if s not in self.answers]
        if missing:
            raise StepProblem(f"Finish this first: {', '.join(missing)}.")

    def districts_frame(self) -> gpd.GeoDataFrame:
        """The saved district file."""
        return _read_unique(self.answers["districts"]["file"])

    def district_ids(self) -> pd.Series:
        """The district IDs of the saved district file, as clean labels."""
        return _labels(self.districts_frame()[self.answers["districts"]["id"]])

    def _match(self, ids: pd.Series, what: str) -> None:
        """Stop unless the IDs of a table match the district IDs exactly (names the differences)."""
        known, given = set(self.district_ids()), set(_labels(ids))
        extra, absent = sorted(given - known)[:6], sorted(known - given)[:6]
        if extra:
            raise StepProblem(
                f"{what}: these rows match no district: {', '.join(extra)}. City-wide total rows must be removed; "
                "if the IDs are written differently (for example 01 and 1), make them identical."
            )
        if absent:
            raise StepProblem(f"{what}: no row for these districts: {', '.join(absent)}.")

    # one method per section
    def set_districts(self, file: str, id_column: str, name_column: str, epsg: int) -> str:
        """Check and save the district file; return a summary for the reader."""
        info = pyogrio.read_info(file)
        if info["crs"] is None:
            raise StepProblem("This file has no coordinate system. Set one in GIS software, then try again.")
        frame = _read_unique(file)
        if frame[id_column].duplicated().any():
            raise StepProblem(f"Column {id_column} has repeated values. District IDs must be unique.")
        if frame.geometry.is_empty.any() or frame.geometry.isna().any():
            raise StepProblem("Some districts have no shape.")
        self.save(
            "districts",
            file=str(file),
            columns=adapters.unique_names(info["fields"]),
            id=id_column,
            name=name_column,
            epsg=int(epsg),
        )
        sample = frame.head(3)
        return (
            f"{len(frame)} districts found. IDs look like {', '.join(_labels(sample[id_column]))}; names look like "
            f"{', '.join(_labels(sample[name_column]))}. "
            "If these look wrong, change the columns above and save again. The IDs must match the IDs in your other "
            "tables."
        )

    def set_network_file(self, file: str, date: str = "") -> str:
        """Check and save a saved Overpass response as the walking network."""
        self.need("districts")
        try:
            network = json.loads(Path(file).read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise StepProblem("This is not a JSON file. Overpass results must be exported as raw data.") from None
        if not isinstance(network, dict) or "elements" not in network:
            raise StepProblem("This is not an Overpass result.")
        figures = osm_download.check_network(network, osm_download.study_area(self.districts_frame(), 0))
        date = date or network.get("osm3s", {}).get("timestamp_osm_base", "")
        if not date:
            raise StepProblem("This file does not say when its map data date from. Type the date.")
        self.save("network", file=str(file), date=date)
        return f"{figures['ways']:,} streets and paths, map data of {date}."

    def download_network(self, say: Callable[[str], None] = print) -> str:
        """Download the walking network from OpenStreetMap in parts and save it."""
        self.need("districts")
        target = self.folder / "osm" / "highways.json"
        figures = osm_download.download_network(self.districts_frame(), target, say)
        self.save("network", file=str(target), date=figures["date"])
        return f"{figures['ways']:,} streets and paths, map data of {figures['date']}."

    def set_facilities(self, file: str, id_column: str = "") -> str:
        """Check and save the existing facilities (they must lie in the districts)."""
        self.need("districts")
        frame = _read_unique(file)
        if frame.crs is None:
            raise StepProblem("This file has no coordinate system.")
        inside = frame.to_crs(4326).representative_point().within(self.districts_frame().to_crs(4326).union_all()).sum()
        if not inside:
            raise StepProblem(
                "None of these facilities lie inside the districts. Check the file and its coordinate system."
            )
        self.save(
            "facilities",
            file=str(file),
            columns=_fields(file),
            id=id_column,
        )
        return f"{len(frame)} facilities, {inside} of them inside the districts."

    def set_demand(self, mode: str) -> str:
        """Save the demand method; return its description."""
        self.save("demand", mode=mode)
        return DEMAND[mode]

    def set_population(
        self,
        file: str,
        id_column: str,
        age_0_5: str,
        age_0_2: str,
        age_3_5: str,
        count_type: str,
        year: int,
        source: str,
    ) -> str:
        """Check and save the table of children per district."""
        self.need("districts", "demand")
        frame, _, _ = wizard._read_table(Path(file))
        used = [age_0_5] if age_0_5 else [age_0_2, age_3_5]
        counts = frame[used].apply(pd.to_numeric, errors="coerce")
        if counts.isna().any().any():
            raise StepProblem(f"{', '.join(used)} must hold plain numbers in every row.")
        self._match(frame[id_column], "Children per district")
        self.save(
            "population",
            file=str(file),
            columns=list(frame.columns),
            id=id_column,
            age_0_5=age_0_5,
            age_0_2=age_0_2,
            age_3_5=age_3_5,
            count_type=count_type,
            year=int(year),
            source=source,
        )
        return f"{len(frame)} districts, {counts.sum().sum():,.0f} children aged 0-5."

    def census_members(self, file: str) -> list[str]:
        """CSV tables of a census file (the file itself, or the CSV members of a zip)."""
        if Path(file).suffix.lower() != ".zip":
            return [Path(file).name]
        with zipfile.ZipFile(file) as archive:
            return [m for m in archive.namelist() if m.lower().endswith(".csv")]

    def census_columns(self, file: str, member: str) -> list[str]:
        """Column names of a census table, read from its header line."""
        with self._census_text(file, member) as text:
            head = text.readline()
        return [c.strip('"') for c in head.strip().split(_separator(head))]

    @contextlib.contextmanager
    def _census_text(self, file: str, member: str) -> Iterator[io.TextIOWrapper]:
        """The census table as text: the file itself, or ``member`` of a zip file (closed afterwards)."""
        with contextlib.ExitStack() as stack:
            if Path(file).suffix.lower() == ".zip":
                raw = stack.enter_context(stack.enter_context(zipfile.ZipFile(file)).open(member))
            else:
                raw = stack.enter_context(Path(file).open("rb"))
            yield stack.enter_context(io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace"))

    def set_census(
        self,
        file: str,
        member: str,
        id_column: str,
        x_column: str,
        y_column: str,
        count_columns: list[str],
        epsg: int,
        cell_size: float,
        year: int,
    ) -> str:
        """Keeps the cells around the city, in a zip with one table, whatever the original file looks like."""
        self.need("districts", "demand", "population")
        columns = self.census_columns(file, member)
        with self._census_text(file, member) as text:
            separator = _separator(text.readline())
        # bounding box of the districts, widened by the margin on every side (minx, miny, maxx, maxy)
        margin = np.array([-CENSUS_MARGIN_M, -CENSUS_MARGIN_M, CENSUS_MARGIN_M, CENSUS_MARGIN_M])
        box = self.districts_frame().to_crs(int(epsg)).total_bounds + margin
        parts = []
        with self._census_text(file, member) as text:
            for chunk in pd.read_csv(text, sep=separator, chunksize=1_000_000, dtype=str):
                x = pd.to_numeric(chunk[x_column], errors="coerce")
                y = pd.to_numeric(chunk[y_column], errors="coerce")
                parts.append(chunk[(x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])])
        cells = pd.concat(parts)
        if cells.empty:
            raise StepProblem("No grid cells lie near the city. Check the coordinate columns and the EPSG code.")
        prepared = self.prepared / "census_grid.zip"
        with zipfile.ZipFile(prepared, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("census_grid.csv", cells.to_csv(sep=";", index=False))
        self.save(
            "census",
            file=str(prepared),
            columns=columns,
            id=id_column,
            x=x_column,
            y=y_column,
            counts=count_columns,
            epsg=int(epsg),
            cell=float(cell_size),
            year=int(year),
            original=str(file),
        )
        return (
            f"{len(cells):,} grid cells kept around the city, saved in {prepared.relative_to(PROJECT_ROOT).as_posix()}."
        )

    def set_direct(self, file: str, id_column: str, children: str, count_type: str, year: int, source: str) -> str:
        """Check and save a map file of child locations."""
        self.need("demand")
        info = pyogrio.read_info(file)
        if info["crs"] is None:
            raise StepProblem("This file has no coordinate system.")
        self.save(
            "direct",
            file=str(file),
            columns=adapters.unique_names(info["fields"]),
            id=id_column,
            children=children,
            count_type=count_type,
            year=int(year),
            source=source,
        )
        return f"{info['features']:,} child locations."

    def set_sites(self, source: str, file: str = "", id_column: str = "") -> str:
        """Save where new facilities can go."""
        self.need("demand")
        columns = _fields(file) if source == "file" else []
        self.save("sites", source=source, file=str(file), columns=columns, id=id_column)
        return (
            "New facilities can go at residential buildings."
            if source == "residential_buildings"
            else f"Sites from {Path(file).name}."
        )

    def needs_buildings(self) -> bool:
        """Whether the building section is needed for the choices made so far."""
        return self.answers["demand"]["mode"] != "direct" or self.answers["sites"]["source"] == "residential_buildings"

    def building_columns(self, file: str, layer: str) -> list[str]:
        """Column names of one layer of the building file."""
        return _fields(file, layer=layer)

    def building_values(self, file: str, layer: str, use: str) -> list[str]:
        """The distinct values of a column of the building file (to pick residential uses)."""
        return sorted(_column_values(file, layer, use).dropna().astype(str).unique())

    def set_buildings(
        self,
        file: str,
        layer: str,
        id_column: str,
        use: str,
        residential: list[str],
        mixed: list[str],
    ) -> str:
        """Check and save the building file and its residential uses."""
        self.need("sites")
        info = pyogrio.read_info(file, layer=layer)
        if "Point" in str(info["geometry_type"]):
            raise StepProblem("These are points. Building footprints (shapes) are needed.")
        count = int(info["features"])
        if use:
            if not residential:
                raise StepProblem("Choose at least one value that means residential.")
            uses = _column_values(file, layer, use).astype(str)
            count = int(uses.isin(residential).sum())
        layers = [name for name, _ in pyogrio.list_layers(file)]
        self.save(
            "buildings",
            file=str(file),
            layers=layers,
            layer=layer,
            columns=[c for c in _read_unique(file, layer=layer, rows=1).columns if c != "geometry"],
            id=id_column,
            use=use,
            residential=residential,
            mixed=mixed,
        )
        note = (
            " This is a large city: expect a build of hours and a solve needing a lot of memory."
            if count > LARGE_CITY
            else ""
        )
        return f"{count:,} residential buildings.{note}"

    def set_priority(self, mode: str, file: str, id_column: str, **fields: Any) -> str:
        """Check and save the table that defines the priority group."""
        self.need("districts")
        frame, _, _ = wizard._read_table(Path(file))
        if mode != "priority_districts":
            self._match(frame[id_column], "Priority table")
        elif not set(_labels(frame[id_column])) <= set(self.district_ids()):
            raise StepProblem("Some listed districts are not in the district file.")
        self.save(
            "priority",
            mode=mode,
            file=str(file),
            columns=list(frame.columns),
            id=id_column,
            **fields,
        )
        if mode == "priority_districts":
            return f"{len(frame)} priority districts."
        if mode == "external_index" and fields["higher"]:
            values = pd.to_numeric(frame[fields["value"]], errors="coerce")
            if values.isna().any():
                raise StepProblem(f"{fields['value']} must hold plain numbers in every row.")
            threshold = values.quantile(fields["quantile"])
            return (
                f"Cut-off {threshold:.2f}: {int((values >= threshold).sum())} of {len(frame)} districts selected: "
                f"{', '.join(_labels(frame[id_column])[values >= threshold])}."
            )
        return f"{len(frame)} districts."

    # settings
    def sequence(self) -> list[str]:
        """The saved answers, in the order the questions of ``city_wizard.create_city`` are asked."""
        a = self.answers
        d, f, mode, s = a["districts"], a["facilities"], a["demand"]["mode"], a["sites"]
        answers = [
            self.name,
            d["file"],
            _number(d["columns"], d["id"]),
            _number(d["columns"], d["name"]),
            str(d["epsg"]),
            "y",
            "1",
            a["network"]["file"],
            a["network"]["date"],
            f["file"],
            _number(f["columns"], f["id"]) if f["id"] else "",
            _number(list(DEMAND), mode),
        ]
        if mode == "census_hybrid":  # asked before the children per district
            c = a["census"]
            answers += [
                "1",  # "Do you have this file?": yes
                c["file"],
                _number(c["columns"], c["id"]),
                _number(c["columns"], c["x"]),
                _number(c["columns"], c["y"]),
                ",".join(_number(c["columns"], n) for n in c["counts"]),
                str(c["epsg"]),
                str(c["cell"]),
                str(c["year"]),
            ]
        if mode != "direct":
            p = a["population"]
            answers += [p["file"], _number(p["columns"], p["id"])]
            answers += (
                ["1", _number(p["columns"], p["age_0_5"])]
                if p["age_0_5"]
                else ["2", _number(p["columns"], p["age_0_2"]), _number(p["columns"], p["age_3_5"])]
            )
            answers += ["1" if p["count_type"] == "observed" else "2", str(p["year"]), p["source"]]
        if mode == "direct":
            c = a["direct"]
            answers += [
                c["file"],
                _number(c["columns"], c["id"]),
                _number(c["columns"], c["children"]),
                "1" if c["count_type"] == "observed" else "2",
                str(c["year"]),
                c["source"],
            ]
        answers.append("1" if s["source"] == "residential_buildings" else "2")
        if s["source"] == "file":
            answers += [s["file"], _number(s["columns"], s["id"])]
        if self.needs_buildings():
            b = a["buildings"]
            answers.append(b["file"])
            if len(b["layers"]) > 1:
                answers.append(_number(b["layers"], b["layer"]))
            answers.append(_number(b["columns"], b["id"]) if b["id"] else "")
            if b["use"]:
                kinds = self.building_values(b["file"], b["layer"], b["use"])
                chosen = [k for k in kinds if k in b["residential"]]
                answers += [
                    "2",
                    _number(b["columns"], b["use"]),
                    ",".join(_number(kinds, k) for k in chosen),
                    ",".join(_number(chosen, k) for k in chosen if k in b["mixed"]),
                ]
            else:
                answers.append("1")
        p = a["priority"]
        answers += [_number(list(PRIORITY), p["mode"]), p["file"], _number(p["columns"], p["id"])]
        if p["mode"] == "four_indicator":
            answers += [_number(p["columns"], p["indicators"][key]) for key in wizard.INDICATORS] + [
                str(p["year"]),
                p["source"],
                str(p["quantile"]),
            ]
        elif p["mode"] == "external_index":
            answers += [
                _number(p["columns"], p["value"]),
                p["name"],
                p["definition"],
                "1" if p["higher"] else "2",
                str(p["quantile"]),
            ]
        else:
            answers += [p["label"], p["definition"]]
        return [*answers, "y"]

    def required(self) -> list[str]:
        """The sections the chosen demand method needs, in order."""
        mode = self.answers.get("demand", {}).get("mode", "census_hybrid")
        steps = ["districts", "network", "facilities", "demand", "sites", "priority"]
        steps += {
            "census_hybrid": ["population", "census"],
            "building_area": ["population"],
            "direct": ["direct"],
        }[mode]
        if "sites" in self.answers and self.needs_buildings():
            steps.append("buildings")
        return steps

    def create(self) -> tuple[Path, list[str]]:
        """Write ``config/<city>.toml`` from the saved answers and check it.

        Replaces only a file made by these setups.
        """
        self.need(*self.required())
        path = PROJECT_ROOT / "config" / f"{self.slug}.toml"
        before = path.read_text(encoding="utf-8") if path.is_file() else None
        if before is not None and not before.startswith("# Created by the question-by-question setup"):
            raise StepProblem(f"{path.name} already exists and was not made here. Choose another city name.")
        answers = iter(self.sequence())
        path.unlink(missing_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            created = wizard.create_city(lambda prompt: next(answers))
        if any(True for _ in answers):
            raise StepProblem("The saved answers do not fit the setup questions. Please report this.")
        ok, lines = check_city(created)
        if not ok:
            raise StepProblem(
                "The settings were saved, but the check found problems:\n" + "\n".join(line[:170] for line in lines)
            )
        self.settings_path = created
        changed = before is not None and before != created.read_text(encoding="utf-8")
        inputs = PROJECT_ROOT / "workspace" / "guided" / self.slug / "inputs"
        notes = [f"Settings saved: {created.relative_to(PROJECT_ROOT).as_posix()}"]
        if changed and inputs.is_dir():
            notes.append(
                f"Your settings changed, but inputs built with the old ones exist. Delete {inputs} to build them again."
            )
        return created, notes

    def city(self) -> guide.City:
        """The city for the run sections."""
        if self.kind == "new":
            if self.settings_path is None:
                raise StepProblem("Create the settings in the review section first.")
            return guide.choose_city(str(self.settings_path))
        if not self.kind:
            raise StepProblem("Start with section 1.")
        return guide.choose_city("demo" if self.kind == "demo" else str(self.settings_path))


# widgets
LAYOUT = {"description_width": "230px"}
WIDE = widgets.Layout(width="760px")


def _ok(output: widgets.Output, text: str) -> None:
    """Show a green confirmation in a section's output."""
    with output:
        display(HTML(f"<div style='color:#176b2c'><b>&#10003; Saved.</b> {text}</div>"))


def _problem(output: widgets.Output, text: str) -> None:
    """Show a problem the reader can correct (in red, HTML escaped) in a section's output."""
    with output:
        display(
            HTML(
                "<div style='color:#a31515;white-space:pre-wrap'><b>Please correct this:</b> "
                + text.replace("<", "&lt;")
                + "</div>"
            )
        )


def _guarded(output: widgets.Output, action: Callable[[], object]) -> Callable[..., None]:
    """Run a button's action: problems the reader can correct are shown in the section, not as a traceback."""

    def run(*_: object) -> None:
        output.clear_output()
        try:
            action()
        except (StepProblem, osm_download.DownloadFailed, wizard.WizardStopped) as problem:
            _problem(output, str(problem))
        except Exception as problem:  # a wrong or damaged file should not end in a traceback
            _problem(output, f"This file could not be used: {str(problem)[:300]}")

    return run


def _dropdown(label: str, columns: list[str], hints: tuple = (), optional: bool = False) -> widgets.Dropdown:
    """A column chooser, preselecting the first usable column whose name contains one of ``hints``."""
    options = ([("(none)", "")] if optional else []) + [(c, c) for c in columns]
    usable = [c for c in columns if c.lower() not in TECHNICAL_COLUMNS]
    match = next(
        (c for hint in hints for c in usable if hint in c.lower()),
        (usable or columns or [None])[0] if not optional else "",
    )
    return widgets.Dropdown(description=label, options=options, value=match, style=LAYOUT, layout=WIDE)


def _text(label: str, value: str = "") -> widgets.Text:
    """A wide text box."""
    return widgets.Text(description=label, value=value, style=LAYOUT, layout=WIDE)


def _year(label: str, value: int = 2025) -> widgets.BoundedIntText:
    """A wide year box (1900-2100)."""
    return widgets.BoundedIntText(description=label, value=value, min=1900, max=2100, style=LAYOUT, layout=WIDE)


def facility_box() -> widgets.BoundedIntText:
    """Show a box for the number of new facilities P (whole number >= 1, default 3) and return it.

    The solve cell reads its value when it runs, so change the number before running that cell.
    """
    box = widgets.BoundedIntText(
        description="New facilities (P)",
        value=guide.DEFAULT_P,
        min=1,
        max=1000,
        style=LAYOUT,
        layout=WIDE,
    )
    display(box)
    display(
        HTML(
            "<p style='max-width:760px'>The default is P = 3. A larger P takes longer to solve. Before "
            "solving, the tool checks that your city has enough candidate sites.</p>"
        )
    )
    return box


def _share() -> widgets.BoundedFloatText:
    """The quantile that defines the priority group (default 0.75: the top quarter)."""
    return widgets.BoundedFloatText(
        description="Priority share (quantile)",
        value=0.75,
        min=0.01,
        max=0.99,
        step=0.05,
        style=LAYOUT,
        layout=WIDE,
    )


def _counted() -> widgets.Dropdown:
    """Whether the counts are observed or projected."""
    return widgets.Dropdown(
        description="These counts are",
        options=[("observed", "observed"), ("projected", "projected")],
        style=LAYOUT,
        layout=WIDE,
    )


class FilePicker(widgets.VBox):
    """Browse this computer's folders, or paste a path. The file stays where it is; nothing is uploaded."""

    def __init__(self, suffixes: tuple, start: Path | None = None) -> None:
        self.suffixes, self.current = suffixes, Path(start or PROJECT_ROOT / "data")
        self.path = widgets.Text(
            description="File",
            style=LAYOUT,
            layout=WIDE,
            placeholder="Choose below, or paste the full path here",
        )
        self.list = widgets.Select(rows=9, layout=WIDE, description="Browse", style=LAYOUT)
        self.where = widgets.HTML()
        self.list.observe(self._chosen, "value")
        super().__init__([self.path, self.where, self.list])
        self._show()

    def _show(self) -> None:
        """List the folders and matching files of the current folder."""
        try:
            items = sorted(self.current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            items = []
        folders = [(f"[folder] {p.name}", p) for p in items if p.is_dir() and not p.name.startswith(".")]
        files = [(p.name, p) for p in items if p.is_file() and p.suffix.lower() in self.suffixes]
        self.list.unobserve(self._chosen, "value")
        self.list.options = [("[up one folder]", self.current.parent), *folders, *files]
        self.list.value = None
        self.list.observe(self._chosen, "value")
        self.where.value = f"<small>Looking in {self.current}</small>"

    def _chosen(self, change: dict[str, Any]) -> None:
        """Open a chosen folder, or put a chosen file into the path box."""
        item = change["new"]
        if item is None:
            return
        if item.is_dir():
            self.current = item
            self._show()
        else:
            self.path.value = str(item)

    @property
    def value(self) -> str:
        """The chosen file as an absolute path; stop with a message if there is none."""
        path = Path(self.path.value.strip().strip('"')).expanduser()
        if not self.path.value.strip():
            raise StepProblem("Choose a file first.")
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise StepProblem(f"No file found at {path}.")
        return str(path)


def _map(shapes: gpd.GeoDataFrame, points: gpd.GeoDataFrame | None = None) -> None:
    """Draw the districts, and optionally points on them, as a small map."""
    figure, axis = plt.subplots(figsize=(5, 4.5))
    shapes.to_crs(4326).plot(ax=axis, color="#e8eef5", edgecolor="#4a6a8a", linewidth=0.7)
    if points is not None:
        points.to_crs(4326).representative_point().plot(ax=axis, color="#b4441c", markersize=6)
    axis.set_axis_off()
    display(figure)
    plt.close(figure)


def _section(setup: Setup, step: str, intro: str) -> None:
    """Show a section's introduction and, if it is already saved, what was saved."""
    display(HTML(f"<p style='max-width:760px'>{intro}</p>"))
    saved = setup.answers.get(step)
    if saved:
        display(
            HTML(
                "<div style='color:#176b2c'>&#10003; This section is already saved. Change it below, or go on to the "
                "next section.</div>"
            )
        )


def _file_section(
    setup: Setup,
    step: str,
    intro: str,
    suffixes: tuple,
    read: Reader,
    fields: FieldMaker,
    save: Saver,
    need: tuple[str, ...] = (),
) -> Widgets:
    """The usual section: pick a file, see what it holds, choose its columns, save."""
    setup.need(*need)
    _section(setup, step, intro)
    picker, read_button, form, output = (
        FilePicker(suffixes),
        widgets.Button(description="Read this file", button_style="primary"),
        widgets.VBox(),
        widgets.Output(),
    )
    save_button = widgets.Button(description="Save this section", button_style="success", disabled=True)
    state = {}

    def do_read() -> None:
        state["file"] = picker.value
        state["columns"], preview = read(state["file"])
        state["fields"] = fields(state["columns"])
        form.children = list(state["fields"].values())
        save_button.disabled = False
        with output:
            preview()

    def do_save() -> None:
        text = save(state["file"], {key: w.value for key, w in state["fields"].items()})
        _ok(output, text)

    read_button.on_click(_guarded(output, do_read))
    save_button.on_click(_guarded(output, do_save))
    display(widgets.VBox([picker, read_button, form, save_button, output]))
    return {
        "picker": picker,
        "read": read_button,
        "form": form,
        "save": save_button,
        "output": output,
        "state": state,
    }


def _friendly(section: Callable[..., Widgets | None]) -> Callable[..., Widgets | None]:
    """A section that cannot run yet says why, without a traceback."""

    @functools.wraps(section)
    def run(*args: Any, **kwargs: Any) -> Widgets | None:
        try:
            return section(*args, **kwargs)
        except StepProblem as problem:
            output = widgets.Output()
            display(output)
            _problem(output, str(problem))

    return run


def _map_file(file: str, rows: int | None = None) -> gpd.GeoDataFrame:
    """Read a map file; refuse files without geometry or coordinate system."""
    frame = _read_unique(file, rows=rows)
    if not isinstance(frame, gpd.GeoDataFrame) or frame.crs is None:
        raise StepProblem("This is not a map file with a coordinate system.")
    return frame


def _read_map(file: str) -> tuple[list[str], Preview]:
    """Columns of a map file and a preview of its first rows (for a file section)."""
    frame = _map_file(file, 5)
    return [c for c in frame.columns if c != "geometry"], lambda: display(frame.drop(columns="geometry"))


def _read_table(file: str) -> tuple[list[str], Preview]:
    """Columns of a table file and a preview of its first rows (for a file section)."""
    frame = wizard._read_table(Path(file))[0]
    return list(frame.columns), lambda: display(frame.head())


def start(setup: Setup, choice_given: str | None = None) -> Widgets | None:
    """Section 1: which city. ``choice_given`` ("demo" or a settings path) skips the question."""
    if choice_given:
        setup.kind = "demo" if choice_given == "demo" else "file"
        setup.settings_path = None if choice_given == "demo" else Path(choice_given)
        print(f"Using {choice_given}. Go to section 13.")
        return None
    choice = widgets.RadioButtons(options=[(text, key) for key, text in STARTS.items()], layout=WIDE)
    name = _text("City name")
    picker = FilePicker((".toml",), PROJECT_ROOT / "config")
    button, output = (
        widgets.Button(description="Continue", button_style="primary"),
        widgets.Output(),
    )

    def go() -> None:
        setup.kind = choice.value
        if choice.value == "new":
            setup.open(name.value)
            saved = ", ".join(setup.answers) or "nothing yet"
            _ok(
                output,
                f"{setup.name}. Answers are kept in {setup.answers_file.relative_to(PROJECT_ROOT).as_posix()}. Saved "
                f"so far: {saved}. Run the next section.",
            )
        elif choice.value == "file":
            setup.settings_path = Path(picker.value)
            _ok(output, "Go to section 13.")
        else:
            _ok(output, "Go to section 13.")

    button.on_click(_guarded(output, go))
    display(
        widgets.VBox(
            [
                choice,
                name,
                widgets.HTML("For a finished settings file only:"),
                picker,
                button,
                output,
            ]
        )
    )
    return {"choice": choice, "name": name, "picker": picker, "button": button, "output": output}


@_friendly
def districts(setup: Setup) -> Widgets | None:
    """Section 2: the district boundaries, with their ID and name columns and a metric EPSG code."""

    def read(file: str) -> tuple[list[str], Preview]:
        frame = _map_file(file)
        state["epsg"] = frame.estimate_utm_crs().to_epsg()
        return [c for c in frame.columns if c != "geometry"], lambda: (
            display(frame.drop(columns="geometry").head()),
            _map(frame),
        )

    def fields(columns: list[str]) -> dict[str, widgets.Widget]:
        return {
            "id": _dropdown("District ID column", columns, ("id", "nr", "code")),
            "name": _dropdown("District name column", columns, ("name", "bezeichnung")),
            "epsg": widgets.BoundedIntText(
                description="Coordinate system in metres (EPSG)",
                value=state["epsg"],
                min=1000,
                max=999999,
                style=LAYOUT,
                layout=WIDE,
            ),
        }

    state = {}
    return _file_section(
        setup,
        "districts",
        "<b>Districts</b> are the areas the city is divided into. The priority group is chosen among them, so use "
        "the smallest areas "
        "that have your child and social data. You need a map file with one shape per district, with an ID and a "
        "name (.gpkg, .geojson or .shp).",
        MAP_FILES,
        read,
        fields,
        lambda file, v: setup.set_districts(file, v["id"], v["name"], v["epsg"]),
    )


@_friendly
def network(setup: Setup) -> Widgets | None:
    """Section 3: the walking network, downloaded from OpenStreetMap or from a saved Overpass file."""
    setup.need("districts")
    _section(
        setup,
        "network",
        "<b>The walking network</b> holds every street and path from OpenStreetMap. Walking times are computed on "
        "it. If you do not have a file, "
        "this section downloads one: it is fetched in parts, checked, and saved, so later runs use the same network. "
        "A large city can take several minutes or more, because the public servers are shared. Messages about busy "
        "servers are normal: the notebook tries others by itself. "
        "Wait for the green tick. Only if you see a red message, click the button again, and it continues from where "
        "it stopped.",
    )
    route = widgets.RadioButtons(
        options=[
            ("Download it for me (needs internet)", "download"),
            ("I already have an Overpass JSON file", "file"),
        ],
        layout=WIDE,
    )
    picker, date = FilePicker((".json",)), _text("Map data date, if asked")
    button, output = (
        widgets.Button(description="Get the network", button_style="primary"),
        widgets.Output(),
    )

    def go() -> None:
        if route.value == "download":
            with output:
                text = setup.download_network()
        else:
            text = setup.set_network_file(picker.value, date.value.strip())
        _ok(output, text)

    button.on_click(_guarded(output, go))
    display(widgets.VBox([route, button, widgets.HTML("For an existing file only:"), picker, date, output]))
    return {"route": route, "picker": picker, "button": button, "output": output}


@_friendly
def facilities(setup: Setup) -> Widgets | None:
    """Section 4: the existing facilities (a map file of points or areas)."""

    def read(file: str) -> tuple[list[str], Preview]:
        frame = _map_file(file)
        return [c for c in frame.columns if c != "geometry"], lambda: (
            display(frame.drop(columns="geometry").head()),
            _map(setup.districts_frame(), frame),
        )

    return _file_section(
        setup,
        "facilities",
        "<b>Existing childcare</b>: the facilities children can already walk to. Use your city's official list, as a "
        "map file of points or shapes. "
        "Results depend on it: a missing Kita makes an area look worse served than it is. Check the map below: the "
        "dots should fall inside your city.",
        MAP_FILES,
        read,
        lambda columns: {"id": _dropdown("Facility ID column", columns, ("id",), optional=True)},
        lambda file, v: setup.set_facilities(file, v["id"]),
        need=("districts",),
    )


@_friendly
def demand(setup: Setup) -> Widgets | None:
    """Section 5: how demand is estimated; decides which of the next sections are needed."""
    setup.need("districts")
    _section(
        setup,
        "demand",
        "<b>How to estimate where children live.</b> This decides the next sections.<br>"
        "&bull; <b>No census grid?</b> Choose <i>district totals and buildings</i>. Children are spread over "
        "residential buildings by floor area, which is less precise within a district.<br>"
        "&bull; The grid is read as a table of cells. Raster files such as .tif cannot be used yet.<br>"
        "&bull; The method changes what results mean, so results from different methods are not directly comparable.",
    )
    mode = widgets.RadioButtons(
        options=[(text, key) for key, text in DEMAND.items()],
        layout=WIDE,
        value=setup.answers.get("demand", {}).get("mode", "census_hybrid"),
    )
    button, output = (
        widgets.Button(description="Save this section", button_style="success"),
        widgets.Output(),
    )
    button.on_click(_guarded(output, lambda: _ok(output, setup.set_demand(mode.value))))
    display(widgets.VBox([mode, button, output]))
    return {"mode": mode, "button": button, "output": output}


@_friendly
def population(setup: Setup) -> Widgets | None:
    """Section 6: children per district (one column for ages 0-5, or two for 0-2 and 3-5)."""
    setup.need("demand")
    if setup.answers["demand"]["mode"] == "direct":
        display(HTML("Not needed: you have child locations. Go on to the next section."))
        return None

    def fields(columns: list[str]) -> dict[str, widgets.Widget]:
        return {
            "id": _dropdown("District ID column", columns, ("id", "nr", "code")),
            "age_0_5": _dropdown("Children 0-5, if one column", columns, ("0_5", "0-5"), optional=True),
            "age_0_2": _dropdown(
                "Children 0-2, if two columns",
                columns,
                ("0_2", "0-2", "unter_3", "unter3"),
                optional=True,
            ),
            "age_3_5": _dropdown("Children 3-5, if two columns", columns, ("3_5", "3-5", "3_bis"), optional=True),
            "count_type": _counted(),
            "year": _year("They refer to the year"),
            "source": _text("Source (publisher, table, date)"),
        }

    def save(file: str, v: dict[str, Any]) -> str:
        if not v["age_0_5"] and not (v["age_0_2"] and v["age_3_5"]):
            raise StepProblem("Choose one column for ages 0-5, or two columns for 0-2 and 3-5.")
        if not v["source"].strip():
            raise StepProblem("Say where the counts come from.")
        return setup.set_population(file, **{"id_column": v["id"], **{k: x for k, x in v.items() if k != "id"}})

    return _file_section(
        setup,
        "population",
        "<b>Children per district</b>: a table with one row per district. Save Excel files as CSV first. Remove any "
        "city-wide total row.",
        TABLES,
        _read_table,
        fields,
        save,
        need=("districts",),
    )


@_friendly
def census(setup: Setup) -> Widgets | None:
    """Section 7: the census grid of children; only the cells around the city are kept."""
    setup.need("demand")
    if setup.answers["demand"]["mode"] != "census_hybrid":
        display(HTML("Not needed for the method you chose. Go on to the next section."))
        return None
    state = {}

    def read(file: str) -> tuple[list[str], Preview]:
        members = setup.census_members(file)
        if not members:
            raise StepProblem("This zip contains no CSV table.")
        state["members"] = members
        state["file"] = file
        choice = next((m for m in members if "100m" in m), members[0])
        state["member"] = choice
        columns = setup.census_columns(file, choice)
        return columns, lambda: display(
            HTML(
                f"Reading table <b>{Path(choice).name}</b>. {len(members)} table(s) in this file. Only the cells "
                "around your city are kept."
            )
        )

    def fields(columns: list[str]) -> dict[str, widgets.Widget]:
        return {
            "member": widgets.Dropdown(
                description="Table",
                options=[(Path(m).name, m) for m in state["members"]],
                value=state["member"],
                style=LAYOUT,
                layout=WIDE,
            ),
            "id": _dropdown("Cell ID column", columns, ("gitter", "id")),
            "x": _dropdown("Cell-centre x column", columns, ("x_mp", "x_")),
            "y": _dropdown("Cell-centre y column", columns, ("y_mp", "y_")),
            "counts": widgets.SelectMultiple(
                description="Columns with children 0-5",
                options=columns,
                rows=8,
                style=LAYOUT,
                layout=WIDE,
            ),
            "epsg": widgets.BoundedIntText(
                description="EPSG of the coordinates",
                value=3035,
                min=1000,
                max=999999,
                style=LAYOUT,
                layout=WIDE,
            ),
            "cell": widgets.BoundedFloatText(
                description="Cell size in metres",
                value=100,
                min=1,
                max=100000,
                style=LAYOUT,
                layout=WIDE,
            ),
            "year": _year("Census year", 2022),
        }

    def save(file: str, v: dict[str, Any]) -> str:
        if not v["counts"]:
            raise StepProblem("Choose the columns with children. Together they must cover ages 0-5.")
        return setup.set_census(
            file,
            v["member"],
            v["id"],
            v["x"],
            v["y"],
            list(v["counts"]),
            v["epsg"],
            v["cell"],
            v["year"],
        )

    return _file_section(
        setup,
        "census",
        "<b>Census grid</b>: children per grid cell, as a table with cell-centre coordinates. "
        "A zip or a CSV is fine, including the official zip for a whole country: this section keeps the cells around "
        "your city. This can take a minute.",
        (".zip", ".csv"),
        read,
        fields,
        save,
        need=("districts", "population"),
    )


@_friendly
def direct(setup: Setup) -> Widgets | None:
    """Section 8: child locations given directly (only for that demand method)."""
    setup.need("demand")
    if setup.answers["demand"]["mode"] != "direct":
        display(HTML("Not needed for the method you chose. Go on to the next section."))
        return None

    def fields(columns: list[str]) -> dict[str, widgets.Widget]:
        return {
            "id": _dropdown("Location ID column", columns, ("id",)),
            "children": _dropdown("Children 0-5 column", columns, ("child", "kind")),
            "count_type": _counted(),
            "year": _year("They refer to the year"),
            "source": _text("Source (register, date)"),
        }

    return _file_section(
        setup,
        "direct",
        "<b>Child locations</b>: a map file of points, each with a number of children.",
        MAP_FILES,
        _read_map,
        fields,
        lambda file, v: setup.set_direct(file, v["id"], v["children"], v["count_type"], v["year"], v["source"]),
    )


@_friendly
def sites(setup: Setup) -> Widgets | None:
    """Section 9: where new facilities can go (residential buildings or a file of candidate sites)."""
    setup.need("demand")
    _section(
        setup,
        "sites",
        "<b>Where new facilities can go.</b> By default, any residential building can be a site. This does not "
        "check whether a site is really available.",
    )
    source = widgets.RadioButtons(
        options=[
            ("At residential buildings (the default)", "residential_buildings"),
            ("At sites listed in a file", "file"),
        ],
        layout=WIDE,
    )
    picker, id_column = FilePicker(MAP_FILES), _text("Site ID column, for a file")
    button, output = (
        widgets.Button(description="Save this section", button_style="success"),
        widgets.Output(),
    )

    def go() -> None:
        if source.value == "file":
            columns = list(pyogrio.read_info(picker.value)["fields"])
            if id_column.value not in columns:
                raise StepProblem(f"Type one of these columns: {', '.join(columns)}.")
            _ok(output, setup.set_sites("file", picker.value, id_column.value))
        else:
            _ok(output, setup.set_sites("residential_buildings"))

    button.on_click(_guarded(output, go))
    display(widgets.VBox([source, button, widgets.HTML("For a file of sites only:"), picker, id_column, output]))
    return {"source": source, "button": button, "output": output}


@_friendly
def buildings(setup: Setup) -> Widgets | None:
    """Section 10: the building file and which of its values mean residential."""
    setup.need("sites")
    if not setup.needs_buildings():
        display(HTML("Not needed for the choices you made. Go on to the next section."))
        return None
    state = {}

    def read(file: str) -> tuple[list[str], Preview]:
        layers = [name for name, _ in pyogrio.list_layers(file)]
        state["file"], state["layers"] = file, layers
        columns = setup.building_columns(file, layers[0])
        return columns, lambda: display(
            HTML(
                f"Layers: {', '.join(layers)}. Using <b>{layers[0]}</b>. Reading the values of a column can take a "
                "moment."
            )
        )

    def fields(columns: list[str]) -> dict[str, widgets.Widget]:
        use = _dropdown(
            "Column saying what each building is",
            columns,
            ("use", "funk", "function", "type", "building"),
            optional=True,
        )
        residential = widgets.SelectMultiple(
            description="Values meaning residential", options=[], rows=10, style=LAYOUT, layout=WIDE
        )
        mixed = widgets.SelectMultiple(
            description="Of those, mixed use with housing",
            options=[],
            rows=6,
            style=LAYOUT,
            layout=WIDE,
        )

        def values(_: object = None) -> None:
            residential.options = (
                setup.building_values(state["file"], state["layers"][0], use.value) if use.value else []
            )

        def chosen(_: object = None) -> None:
            mixed.options = list(residential.value)

        use.observe(values, "value")
        residential.observe(chosen, "value")
        values()
        return {
            "layer": widgets.Dropdown(description="Layer", options=state["layers"], style=LAYOUT, layout=WIDE),
            "id": _dropdown("Unique building ID column", columns, ("id",), optional=True),
            "use": use,
            "residential": residential,
            "mixed": mixed,
        }

    return _file_section(
        setup,
        "buildings",
        "<b>Buildings</b>: footprints (shapes, not points). They show where people live and, by default, where new "
        "facilities can go. "
        "Leave the use column at (none) only if every building in the file is residential.",
        MAP_FILES,
        read,
        fields,
        lambda file, v: setup.set_buildings(
            file, v["layer"], v["id"], v["use"], list(v["residential"]), list(v["mixed"])
        ),
    )


@_friendly
def priority(setup: Setup) -> Widgets | None:
    """Section 11: the priority group, from the default index, a documented index or a fixed list."""
    setup.need("districts")
    _section(
        setup,
        "priority",
        "<b>The priority group</b>: which districts count as most vulnerable. The Rawlsian models protect this group. "
        "The default index needs four columns, each measuring the concept named for it.",
    )
    mode = widgets.RadioButtons(options=[(text, key) for key, text in PRIORITY.items()], layout=WIDE)
    picker, read_button, form, output = (
        FilePicker(TABLES),
        widgets.Button(description="Read this file", button_style="primary"),
        widgets.VBox(),
        widgets.Output(),
    )
    save_button = widgets.Button(description="Save this section", button_style="success", disabled=True)
    state = {}

    def do_read() -> None:
        state["file"] = picker.value
        columns, preview = _read_table(state["file"])
        made = {"id": _dropdown("District ID column", columns, ("id", "nr", "code"))}
        if mode.value == "four_indicator":
            made.update({key: _dropdown(label, columns) for key, label in wizard.INDICATORS.items()})
            made.update(
                year=_year("Year of these values"),
                source=_text("Source and definitions"),
                quantile=_share(),
            )
        elif mode.value == "external_index":
            made.update(
                value=_dropdown("Index column", columns),
                name=_text("Name and year of the index"),
                definition=_text("Publisher and method"),
                higher=widgets.Checkbox(
                    description="A higher value means more vulnerable",
                    value=True,
                    style=LAYOUT,
                    layout=WIDE,
                ),
                quantile=_share(),
            )
        else:
            made.update(label=_text("Name of the list"), definition=_text("Source and selection rule"))
        state["fields"] = made
        form.children = list(made.values())
        save_button.disabled = False
        with output:
            preview()

    def do_save() -> None:
        v = {key: w.value for key, w in state["fields"].items()}
        id_column = v.pop("id")
        if any(isinstance(x, str) and not x.strip() for x in v.values()):
            raise StepProblem("Please fill in every box.")
        if mode.value == "four_indicator":
            indicators = {key: v.pop(key) for key in wizard.INDICATORS}
            v["indicators"] = indicators
        _ok(output, setup.set_priority(mode.value, state["file"], id_column, **v))

    read_button.on_click(_guarded(output, do_read))
    save_button.on_click(_guarded(output, do_save))
    display(widgets.VBox([mode, picker, read_button, form, save_button, output]))
    return {
        "mode": mode,
        "picker": picker,
        "read": read_button,
        "form": form,
        "save": save_button,
        "output": output,
    }


@_friendly
def review(setup: Setup) -> Widgets | None:
    """Everything chosen, then the settings file. Nothing heavy happens until the reader confirms."""
    if setup.kind != "new":
        display(HTML("Not needed: go on to the next section."))
        return None
    missing = [s for s in setup.required() if s not in setup.answers]
    rows = [
        (step, str(values.get("file") or values.get("mode") or values.get("source") or ""))
        for step, values in setup.answers.items()
        if step in setup.required()
    ]
    display(pd.DataFrame(rows, columns=["Section", "Choice"]).style.hide(axis="index"))
    if missing:
        raise StepProblem(f"Still to do: {', '.join(missing)}.")
    button, output = (
        widgets.Button(description="Create the settings", button_style="success"),
        widgets.Output(),
    )

    def go() -> None:
        _, notes = setup.create()
        _ok(output, "<br>".join(notes) + "<br>Run the next section.")

    button.on_click(_guarded(output, go))
    display(widgets.VBox([button, output]))
    return {"button": button, "output": output}
