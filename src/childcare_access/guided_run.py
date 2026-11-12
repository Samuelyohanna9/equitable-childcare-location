"""Guided run for the notebook ``quickstart.ipynb``: the demo city or any configured city.

This module only drives the steps. The analysis runs through the command-line workflow
(``childcare-access check | build | solve``) in separate processes, so the formulations and solver settings are those
of the package. This module adds the choice of
city and of the number of new facilities P, the licence step, a live report and a summary of the results.

Licence handling: entered secrets stay in this Python process, are never printed or written to the notebook, and reach
Gurobi through a temporary licence file that exists only during the run. Gurobi must receive the real values, so a
secret cannot be replaced by a hash; the hash here is only a salted fingerprint for the run record.
"""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import os
import pickle
import queue
import re
import secrets as _secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .config import CONFIG_ENV, PROJECT_ROOT, check_city

DEFAULT_P = 3  # default number of new facilities; the user may choose another
LICENCE_LINE = re.compile(
    r"(?i)licen[sc]e\s*id|licenseid|academic license|named-user|wlsaccessid|wlssecret|registered to|hostid|"
    r"gurobi\.lic|license expires|cloud license|wls license|parameter username"
)
EMAIL = re.compile(r"[\w.%+-]+@[\w-]+(\.[\w-]+)+")
SIZE_LIMITED = 10010
MARKERS = (b"licenseid", b"academic license", b"wlsaccessid", b"wlssecret", b"registered to")


class GuidedRunError(RuntimeError):
    """A problem the user can act on; the message says what to do.

    In a notebook it is shown as plain text with one short red line, not as a Python traceback.
    """

    def _render_traceback_(self) -> list[str]:
        """How IPython shows this error: the message as normal output, then one line saying the cell stopped.

        IPython may ask more than once; the message is printed only the first time.
        """
        if not getattr(self, "_shown", False):
            self._shown = True
            print(str(self))
        return ["\x1b[1;31mStopped here. The message above says what to do.\x1b[0m"]


@dataclass
class City:
    """A city to run: its settings file, workspace, inputs folder and the expected solver time."""

    name: str
    config: Path
    workspace: Path
    inputs: Path
    expected: str


@dataclass
class Licence:
    """How Gurobi gets its licence, with a salted fingerprint; entered keys are kept out of its repr."""

    method: str  # "installed", "file" or "wls"
    path: Path | None = None
    fingerprint: str = ""
    _values: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return f"Licence(method={self.method!r}, fingerprint={self.fingerprint!r})"

    def secrets(self) -> list[str]:
        """The entered keys, to be hidden from every line shown or saved."""
        return [v for v in self._values.values() if v]


def _clock(seconds: float) -> str:
    """Elapsed time as ``1h 02m 03s`` (or `` 2m 03s`` below an hour)."""
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {rest:02d}s" if hours else f"{minutes:2d}m {rest:02d}s"


def redact(line: str, secrets: list[str] | tuple = ()) -> str:
    """A line that is safe to show: licence lines hidden, e-mail addresses and entered secrets removed."""
    for value in secrets:
        line = line.replace(value, "[hidden]")
    if LICENCE_LINE.search(line):
        return "[licence line hidden]"
    return EMAIL.sub("[e-mail hidden]", line).replace(str(PROJECT_ROOT), "<project>").replace(str(Path.home()), "~")


# 1. city and data
TEMPLATE = PROJECT_ROOT / "config" / "city_template.toml"


def sample_settings(name: str) -> Path:
    """Create ``config/<city>.toml`` from the template, with the city name filled in, and return its path.

    An existing file is never replaced: it is returned as it is, so a half-filled file keeps its values.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise GuidedRunError("Please type a city name. Run this cell again.")
    target = PROJECT_ROOT / "config" / f"{slug}.toml"
    if target.exists():
        return target
    text = TEMPLATE.read_text(encoding="utf-8")
    filled = {
        'name = "Example City"                 # <-- REPLACE': f"name = {json.dumps(name.strip(), ensure_ascii=False)}",
        'scenario = "example_city"             # <-- REPLACE: short name, used for node-file folders': (
            f'scenario = "{slug}"                   # short name, used for node-file folders'
        ),
        'root = "workspace/example_city"       # <-- REPLACE: where build/solve/compare write (git-ignored)': (
            f'root = "workspace/{slug}"           # where build/solve/compare write (git-ignored)'
        ),
    }
    for old, new in filled.items():
        if old not in text:
            raise GuidedRunError(f"{TEMPLATE.name} has changed; copy it by hand to {target.name} and fill it in.")
        text = text.replace(old, new, 1)
    target.write_text(text, encoding="utf-8")
    return target


def _show_file_link(path: Path) -> None:
    """A link that opens ``path`` in JupyterLab's editor (in a notebook), else its path."""
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        get_ipython = None
    if get_ipython is not None and get_ipython() is not None:
        display(
            HTML(
                f'<p style="font-size:1.1em">&#128221; <a href="{relative}" target="_blank"><b>Open {relative}</b></a>'
                " to fill it in (it opens in a new tab here).</p>"
            )
        )
    else:
        print(f"Open this file in a text editor and fill it in: {path}")


def choose_city(choice: str | None = None) -> City:
    """Ask which city to run, unless ``choice`` is given: "demo" or the path of a city configuration."""
    if not choice:
        print(
            "Which city do you want to analyse?\n  1  Demo city Testburg (small synthetic city; about 1 minute)\n"
            "  2  Your own city"
        )
        answer = input("Type 1 or 2: ").strip()
        if answer == "2":
            print(
                "\nDo you already have a settings file for your city?\n  1  No, create one by asking me questions\n  "
                "2  Yes, I will give its path\n  3  No, give me a sample file to fill in myself"
            )
            which = input("Type 1, 2 or 3: ").strip()
            if which == "3":
                target = sample_settings(input("City name: "))
                relative = target.relative_to(PROJECT_ROOT).as_posix()
                print(
                    f"\nYour settings file: {relative}\n"
                    "  1. Open it with the link below.\n"
                    "  2. Replace every value marked  <-- REPLACE . The comment next to each value explains it;\n"
                    "     More help: docs/APPLYING_TO_ANOTHER_CITY.md. Sections of unused methods can stay.\n"
                    "  3. Save it (Ctrl+S).\n"
                    "  4. Run this cell again, choose 2 (own city), then 2 (I will give its path), and type:\n"
                    f"     {relative}\n"
                    "     The data check then lists anything still missing."
                )
                _show_file_link(target)
                raise GuidedRunError(f"Fill in {relative}, save it, then run this cell again (see the steps above).")
            if which == "2":
                choice = input("Path of your city configuration (.toml): ").strip().strip('"')
                if not choice:
                    raise GuidedRunError("No path given. Run this cell again.")
            else:
                from .city_wizard import WizardStopped, create_city

                try:
                    choice = str(create_city())
                except WizardStopped as stopped:
                    raise GuidedRunError(str(stopped)) from None
        else:
            choice = {"1": "demo"}.get(answer)
            if choice is None:
                raise GuidedRunError(f"'{answer}' is not one of the options. Run this cell again and type 1 or 2.")
    root = PROJECT_ROOT / "workspace" / "guided"
    if choice == "demo":
        city = City(
            "Testburg (demo)",
            PROJECT_ROOT / "config" / "synthetic_city.toml",
            root / "demo",
            root / "demo" / "inputs",
            "about 1 minute",
        )
    else:
        path = Path(choice).expanduser()
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.is_file():
            raise GuidedRunError(
                f"Configuration not found: {path}\nCopy config/city_template.toml, fill it in, and give its path. "
                "Guide: docs/APPLYING_TO_ANOTHER_CITY.md"
            )
        city = City(
            path.stem,
            path,
            root / path.stem,
            root / path.stem / "inputs",
            "depends on the size of the city",
        )
    print(f"City: {city.name}. Expected solver time (P = {DEFAULT_P}): {city.expected}.")
    return city


def choose_p(answer: int | str | None = None) -> int:
    """Ask for the number of new facilities P, unless ``answer`` is given; Enter keeps the default P = 3.

    P must be a whole number of at least 1. Whether the city has enough candidates for it is checked before
    solving (the solve step refuses an infeasible P with an explanation).
    """
    if answer is None:
        answer = input(
            f"How many new facilities (P) should the models place? Press Enter for {DEFAULT_P}: "
        ).strip() or str(DEFAULT_P)
    text = str(answer).strip()
    if not text.isdigit() or int(text) < 1:
        raise GuidedRunError(f"'{text}' is not a whole number of at least 1. Run this cell again.")
    p = int(text)
    note = " (the default)" if p == DEFAULT_P else f" (the default is {DEFAULT_P})"
    print(f"P = {p} new facilities{note}.")
    return p


def check_data(city: City) -> None:
    """Confirm that the data for this city are present and usable, and say exactly what is missing."""
    print(f"Checking the data for {city.name} ...")
    if (
        city.name.startswith("Testburg")
        and not (PROJECT_ROOT / "examples" / "synthetic_city" / "data" / "forecast.csv").exists()
    ):
        print("Creating the demo city's data ...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "examples" / "synthetic_city" / "make_city.py")],
            check=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    ok, lines = check_city(city.config)
    for line in lines:
        print(" ", line[:170])
    if not ok:
        raise GuidedRunError(
            "The city data or configuration need attention; the lines above say what. Fix them and run this cell "
            "again.\n"
            "Guide: docs/APPLYING_TO_ANOTHER_CITY.md"
        )
    print("Data ready.")


def build_inputs(city: City) -> None:
    """Build the model inputs for this city. Offline, no Gurobi."""
    if (city.inputs / "validation_report.json").is_file():
        print(f"Skipped: inputs already built in {city.inputs}. Delete that folder to build them again.")
        return
    command = [
        sys.executable,
        "-m",
        "childcare_access",
        "build",
        "--config",
        str(city.config),
        "--workspace",
        str(city.workspace),
    ]
    run_live("Building the model inputs", command, _environment(city))


def _environment(city: City) -> dict:
    """Environment of the solver processes: the city's settings selected, unbuffered UTF-8 output."""
    return {
        **os.environ,
        CONFIG_ENV: str(city.config),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }


# 2. licence
PRIVACY = (
    "How your licence is handled:\n"
    "  - Keys you type here are hidden while typing, and are never printed or saved in this notebook or in the run "
    "results.\n"
    "  - They stay in this notebook's memory. During the solve only, they sit in a temporary file so Gurobi can read "
    "them; it is deleted afterwards.\n"
    "  - Only a salted SHA-256 fingerprint is recorded. It cannot be turned back into your licence.\n"
    "  - Gurobi needs the real licence, so the licence itself cannot be hashed. A web licence (WLS) is sent to "
    "Gurobi's own licence service.\n"
    "  - Anyone with access to this computer while the notebook is running could in principle read its memory. Use "
    "your own machine."
)


def _fingerprint(values: list[str]) -> str:
    """Salted SHA-256 fingerprint of the licence values (cannot be turned back into them)."""
    salt = _secrets.token_hex(16)
    return "sha256:" + hashlib.sha256((salt + "|".join(values)).encode("utf-8")).hexdigest()[:16]


def ask_licence(choice: str | None = None, path: str | None = None) -> Licence:
    """Ask how Gurobi should find its licence, unless ``choice`` is given: "installed", "file" or "wls"."""
    print(PRIVACY)
    if not choice:
        print(
            "\nHow should Gurobi find your licence?\n  1  It is already installed on this computer\n  2  I have a "
            "gurobi.lic file\n"
            "  3  I have a Web License Service (WLS) licence: access ID, secret and licence ID"
        )
        choice = {"1": "installed", "2": "file", "3": "wls"}.get(input("Type 1, 2 or 3: ").strip())
    if choice == "installed":
        licence = Licence("installed")
    elif choice == "file":
        file = Path(path or input("Path of your gurobi.lic file: ").strip().strip('"')).expanduser()
        if not file.is_file():
            raise GuidedRunError(f"No file at {file}. Run this cell again with the correct path.")
        licence = Licence(
            "file",
            path=file,
            fingerprint=_fingerprint([hashlib.sha256(file.read_bytes()).hexdigest()]),
        )
    elif choice == "wls":
        values = {
            "WLSACCESSID": getpass.getpass("WLS access ID (hidden): ").strip(),
            "WLSSECRET": getpass.getpass("WLS secret (hidden): ").strip(),
            "LICENSEID": getpass.getpass("Licence ID (hidden): ").strip(),
        }
        if not all(values.values()):
            raise GuidedRunError("All three WLS values are needed. Run this cell again.")
        licence = Licence("wls", fingerprint=_fingerprint(list(values.values())), _values=values)
    else:
        raise GuidedRunError("That is not one of the options. Run this cell again and type 1, 2 or 3.")
    print(f"\nLicence method: {licence.method}. Fingerprint: {licence.fingerprint or 'none (no key entered)'}.")
    return licence


@contextlib.contextmanager
def _licence_environment(city: City, licence: Licence) -> Iterator[dict[str, str]]:
    """The solver processes' environment. A WLS licence exists as a file only inside this block."""
    environment = _environment(city)
    folder = None
    try:
        if licence.method == "file":
            environment["GRB_LICENSE_FILE"] = str(licence.path)
        elif licence.method == "wls":
            folder = Path(tempfile.mkdtemp(prefix="gurobi_session_"))
            file = folder / "gurobi.lic"
            file.write_text("".join(f"{k}={v}\n" for k, v in licence._values.items()), encoding="utf-8")
            with contextlib.suppress(OSError):
                file.chmod(0o600)
            environment["GRB_LICENSE_FILE"] = str(file)
        if licence.method != "installed":  # a licence in the user folder must not silently win over the one given here
            home = Path(tempfile.mkdtemp(prefix="gurobi_home_"))
            environment.update({"USERPROFILE": str(home), "HOME": str(home), "CHILDCARE_GUIDED_HOME": str(home)})
        yield environment
    finally:
        for key in ("CHILDCARE_GUIDED_HOME",):
            if key in environment:
                shutil.rmtree(environment[key], ignore_errors=True)
        if folder:
            shutil.rmtree(folder, ignore_errors=True)
            print(
                "Temporary licence file deleted."
                if not folder.exists()
                else f"WARNING: please delete {folder} by hand."
            )


CHECK = """
import json
try:
    import gurobipy as gp
except ImportError:
    print("RESULT " + json.dumps({"state": "missing"})); raise SystemExit
try:
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    model = gp.Model(env=env); model.Params.OutputFlag = 0
    x = model.addVars(2100, vtype="B"); model.update(); model.optimize()
    version = ".".join(map(str, gp.gurobi.version())); model.dispose(); env.dispose()
    print("RESULT " + json.dumps({"state": "ok", "version": version}))
except gp.GurobiError as error:
    print("RESULT " + json.dumps({"state": "error", "code": error.errno}))
"""


def check_licence(city: City, licence: Licence) -> None:
    """Start Gurobi once with this licence, and show only the outcome."""
    with _licence_environment(city, licence) as environment:
        run = subprocess.run(
            [sys.executable, "-c", CHECK],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
    lines = [line[7:] for line in run.stdout.splitlines() if line.startswith("RESULT ")]
    outcome = json.loads(lines[-1]) if lines else {"state": "error", "code": "unknown"}
    if outcome["state"] == "missing":
        raise GuidedRunError(
            "Gurobi is not installed in this environment. Run `uv sync --locked` (see README.md) and restart the "
            "notebook."
        )
    if outcome["state"] == "error":
        if outcome["code"] == SIZE_LIMITED:
            raise GuidedRunError(
                "Gurobi started without a full licence: the free pip version only allows small models. Please supply "
                "a full licence."
            )
        raise GuidedRunError(
            f"Gurobi did not accept this licence (error {outcome['code']}). Check it and run the licence cells again. "
            "Free academic licences: https://www.gurobi.com/academia/"
        )
    print(f"Licence accepted by Gurobi {outcome['version']}.")


# 3. live report
def _latest_solver_line(folder: Path | None, secrets: Iterable[str]) -> str:
    """Last line of the newest solver log in ``folder`` (redacted), for the progress line."""
    if not folder or not folder.is_dir():
        return ""
    logs = sorted(folder.rglob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return ""
    with logs[-1].open("rb") as handle:
        handle.seek(max(0, logs[-1].stat().st_size - 4000))
        lines = [line for line in handle.read().decode("utf-8", "replace").splitlines() if line.strip()]
    return f"{logs[-1].parent.name}: {redact(lines[-1].strip(), secrets)[:110]}" if lines else ""


def _forward(stream: Iterable[str], sink: queue.Queue) -> None:
    """Put every line of ``stream`` into ``sink`` (runs in a thread, so reading never blocks the progress lines)."""
    for line in stream:
        sink.put(line)


def run_live(
    title: str,
    command: list[str],
    environment: dict,
    watch: Path | None = None,
    secrets: Iterable[str] = (),
    heartbeat: int = 30,
) -> None:
    """Run a step in its own process and report while it runs.

    Every message is shown as it arrives, and a progress line whenever the process is quiet, so the reader can see
    that it is still working.
    """
    started = time.monotonic()
    print(f"=== {title}: started {datetime.now():%H:%M:%S} ===", flush=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=_forward, args=(process.stdout, lines), daemon=True).start()
    recent, problem, last = [], [], time.monotonic()
    while process.poll() is None or not lines.empty():
        try:
            line = redact(lines.get(timeout=1).rstrip(), secrets)
        except queue.Empty:
            line = ""
        if line:
            recent = [*recent, line][-15:]
            if problem or line.startswith("INPUT PROBLEM:"):  # a short message from the build: keep it all
                problem.append(line.removeprefix("INPUT PROBLEM:").strip())
            print(f"[{_clock(time.monotonic() - started)}] {line[:170]}", flush=True)
            last = time.monotonic()
        elif time.monotonic() - last >= heartbeat:
            detail = _latest_solver_line(watch, secrets)
            print(f"[{_clock(time.monotonic() - started)}] still working ... {detail}", flush=True)
            last = time.monotonic()
    if process.returncode and problem:
        raise GuidedRunError(
            f"{title} stopped: the input data need correcting.\n  "
            + "\n  ".join(problem)
            + "\nCorrect the file or the settings, then run this cell again."
        )
    if process.returncode:
        raise GuidedRunError(
            f"{title} stopped with an error after {_clock(time.monotonic() - started)}. Last messages:\n  "
            + "\n  ".join(recent[-8:])
        )
    print(f"=== {title}: finished in {_clock(time.monotonic() - started)} ===", flush=True)


# 4. solve and results
def _scan(folder: Path, secrets: Iterable[str]) -> list[str]:
    """Saved files that contain a licence marker or an entered secret (normally none)."""
    wanted = [*MARKERS, *(s.encode("utf-8") for s in secrets)]
    return [
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.is_file() and any(w in p.read_bytes().lower() or w in p.read_bytes() for w in wanted)
    ]


def solve(city: City, licence: Licence, p: int = DEFAULT_P) -> Path:
    """Solve the three models for ``p`` new facilities with the package's settings, writing to a new run folder."""
    if not (city.inputs / "inputs_bundle.pkl").is_file():
        raise GuidedRunError("The model inputs are not ready. Run the build step first.")
    name = f"P{p}_{datetime.now():%Y%m%d_%H%M%S}"
    run = city.workspace / "runs" / name
    command = [
        sys.executable,
        "-m",
        "childcare_access",
        "solve",
        "--config",
        str(city.config),
        "--workspace",
        str(city.workspace),
        "--inputs",
        str(city.inputs),
        "--p",
        str(p),
        "--run-name",
        name,
        "--yes",
    ]
    print("You can leave this running. Keep the computer awake and this notebook open; closing it stops the run.")
    with _licence_environment(city, licence) as environment:
        run_live(
            f"Solving the three models for {city.name}",
            command,
            environment,
            watch=run,
            secrets=licence.secrets(),
        )
    found = _scan(run, licence.secrets())
    print(
        "Saved run files checked: no licence text found."
        if not found
        else f"WARNING: licence text found in {found}. Do not share these files."
    )
    record = {
        "city": city.name,
        "P": p,
        "licence_method": licence.method,
        "licence_fingerprint": licence.fingerprint,
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "licence_text_found": found,
    }
    (run / "guided_run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Results saved in {run}")
    return run


def summarise(city: City, run: Path) -> Path:
    """Compute the results in a separate process (the package reads the city configuration on import)."""
    run_live(
        "Computing the results",
        [
            sys.executable,
            "-m",
            "childcare_access.guided_run",
            "summary",
            str(city.inputs),
            str(run),
        ],
        _environment(city),
    )
    return run / "guided_results"


def _summary(inputs: Path, run: Path) -> None:
    """Writes the result tables and maps with the package's own functions."""
    import matplotlib

    matplotlib.use("Agg")

    from . import evaluation as ev
    from . import report
    from . import solutions as sol
    from .comparison import _NoSites

    out = run / "guided_results"
    out.mkdir(exist_ok=True)
    print("Loading the inputs and solutions ...", flush=True)
    with (inputs / "inputs_bundle.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    with (inputs / bundle["network"]["routing_graph"]).open("rb") as handle:
        graph = pickle.load(handle)
    districts, kitas, candidates = bundle["districts"], bundle["kitas"], bundle["candidates"]
    demand = bundle["demand"].set_index("demand_id")
    nodes = candidates.set_index("facility_id")["nearest_node"]
    best = sol.load_reported_solutions(run, config=_NoSites())
    print("Recomputing walking times for the current situation and each model ...", flush=True)
    times = {"current": ev.scenario_times(graph, demand, kitas["nearest_node"], nodes)}
    for key, solution in best.items():
        times[key] = ev.scenario_times(graph, demand, kitas["nearest_node"], nodes, solution.sites)
    names = {"current": "Current", **{k: v[0] for k, v in report.MODELS.items()}}
    indicators = report.indicators(times, demand)
    indicators.rename(columns=names).to_csv(out / "indicators.csv")
    report.relative_change(indicators).mul(100).rename(columns=names).to_csv(out / "change_vs_current_percent.csv")
    report.gains(times, demand).rename(columns=names).to_csv(out / "savings.csv")
    status = report.status_summary(run)[
        [
            "model",
            "role",
            "solver_status",
            "certification",
            "objective",
            "best_bound",
            "mip_gap",
            "runtime_s",
            "time_limit_s",
        ]
    ]
    status.to_csv(out / "certification.csv", index=False)
    import pandas as pd

    pd.DataFrame(
        [
            {
                "model": names[k],
                "sites": " / ".join(map(str, s.sites)),
                "objective": s.objective_value,
            }
            for k, s in best.items()
        ]
    ).to_csv(out / "sites.csv", index=False)
    print("Drawing the maps ...", flush=True)
    report.sites_map(districts, kitas, candidates, best).savefig(out / "sites_map.png", dpi=130, bbox_inches="tight")
    report.savings_grid_maps(districts, bundle["demand"], times, candidates, best).savefig(
        out / "savings_grid_maps.png", dpi=130, bbox_inches="tight"
    )
    report.priority_chart(indicators).savefig(out / "priority_group_walks.png", dpi=130, bbox_inches="tight")
    report.distribution_boxplots(times, demand).savefig(out / "walking_time_boxplots.png", dpi=130, bbox_inches="tight")


def show_results(folder: Path) -> None:
    """Display the saved result tables and maps."""
    import pandas as pd
    from IPython.display import Image, Markdown, display

    pd.set_option("display.max_colwidth", 90)
    status = pd.read_csv(folder / "certification.csv")
    display(
        Markdown(
            "**Solver certification.** Only *proven optimal* results are optimal. A *time-limited* result is a "
            "feasible solution, not a proven optimum."
        )
    )
    display(status)
    display(Markdown("**Selected sites**"))
    display(pd.read_csv(folder / "sites.csv"))
    display(Markdown("**Accessibility (minutes unless stated)**"))
    display(pd.read_csv(folder / "indicators.csv", index_col=0))
    display(Markdown("**Change relative to the current situation (%)**"))
    display(pd.read_csv(folder / "change_vs_current_percent.csv", index_col=0).style.format("{:+.1f}%"))
    display(Markdown("**Savings against the current situation**"))
    display(pd.read_csv(folder / "savings.csv", index_col=0))
    for name in ("sites_map.png", "savings_grid_maps.png"):
        display(Image(filename=str(folder / name)))
    display(
        Markdown(
            "**Walks of the vulnerable group.** The mean, the longest walk of any location, and the highest district "
            "mean, per scenario."
        )
    )
    display(Image(filename=str(folder / "priority_group_walks.png")))
    display(
        Markdown(
            "**Walking-time distribution.** Every demand location, citywide and in the priority group; each "
            "location counts once. The line in a box is the median, the diamond the unweighted mean and the box the "
            "middle 50% of locations; outlier markers are omitted."
        )
    )
    display(Image(filename=str(folder / "walking_time_boxplots.png")))
    print(f"All of this is saved in {folder}")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        _summary(Path(sys.argv[2]), Path(sys.argv[3]))
