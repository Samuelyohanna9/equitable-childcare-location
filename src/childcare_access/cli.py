"""Command line of the package: ``childcare-access check|build|solve|compare``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """The ``childcare-access`` command line and its sub-commands."""
    parser = argparse.ArgumentParser(
        prog="childcare-access",
        description="Locate new childcare facilities for a city: check, build, solve and compare.",
    )
    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser("check", help="Check a city configuration and its input data.")
    check_parser.add_argument("--city", default=None, help="City configuration config/<city>.toml.")
    check_parser.add_argument("--config", default=None, help="Explicit path to a city configuration.")

    for name, text in [
        (
            "build",
            "Build the model inputs from the raw data into the workspace (offline, no solver).",
        ),
        ("solve", "Solve the three models with Gurobi on the built inputs (workspace)."),
        ("compare", "Compare a fresh run with reference results (the [outputs] folders of the configuration)."),
    ]:
        sub = subparsers.add_parser(name, help=text)
        sub.add_argument("--city", default=None, help="City configuration config/<city>.toml.")
        sub.add_argument("--config", default=None, help="Explicit path to a city configuration.")
        sub.add_argument(
            "--workspace",
            default=None,
            help="Workspace folder (default: [workspace] root of the configuration).",
        )
        if name in ("solve", "compare"):
            sub.add_argument("--run-name", default=None, help="Run folder name (default P<budget>).")
        if name == "solve":
            sub.add_argument(
                "--p",
                type=int,
                default=None,
                help="Facility budget (default: [model_parameters] facility_budget).",
            )
            sub.add_argument(
                "--models",
                default="utilitarian,rawlsian,district",
                help="Comma-separated subset of the three models.",
            )
            sub.add_argument("--inputs", default=None, help="Inputs folder (default: <workspace>/inputs).")
            sub.add_argument(
                "--district-warm-start",
                choices=["same-run", "none"],
                default="same-run",
                help=(
                    "Start the District model from the better of this run's Utilitarian and Individual sites "
                    "(default), or none."
                ),
            )
            sub.add_argument("--yes", action="store_true", help="Confirm a large (long) optimisation run.")
        if name == "compare":
            sub.add_argument(
                "--reference-run",
                default=None,
                help="Reference run folder (default: [outputs] final_run of the configuration).",
            )
            sub.add_argument(
                "--reference-inputs",
                default=None,
                help="Reference inputs folder (default: [outputs] final_inputs of the configuration).",
            )
            sub.add_argument(
                "--inputs-only",
                action="store_true",
                help="Compare only the built inputs with the reference inputs.",
            )
            sub.add_argument(
                "--against-inputs",
                default=None,
                help="Classified comparison (A identical inputs / B same city / C different cities) of "
                "<workspace>/inputs (and --run-name) with these inputs (and --against-run).",
            )
            sub.add_argument("--against-run", default=None, help="Run folder of the --against-inputs build.")
    return parser


def _config_path(args: argparse.Namespace) -> Path:
    """The configuration named by ``--config`` or ``--city``; one of them is required."""
    from .config import PROJECT_ROOT

    if args.config:
        return Path(args.config)
    if args.city:
        return PROJECT_ROOT / "config" / f"{args.city}.toml"
    raise SystemExit("Name the city: --city <name> (config/<name>.toml) or --config <path>.")


def _input_problem(problem: Exception, path: Path) -> str:
    """A problem with the input data, as a short message for the user (no traceback)."""
    text = f"{INPUT_PROBLEM} {problem}"
    if "not in the districts" in text or "districts without a row" in text or "not among the districts" in text:
        text += (
            "\nThe district IDs of this table differ from those of the district file (sources.districts.id_column)."
            f"\nUse the same IDs in both: choose the matching columns in {path.name}, or correct the table."
        )
    return text


def _workflow(args: argparse.Namespace) -> None:
    """Select the city configuration BEFORE the modules that read it are imported, then run the workflow."""
    import os

    from .config import CONFIG_ENV

    path = _config_path(args)
    os.environ[CONFIG_ENV] = str(path)
    from . import workflow

    workspace = workflow.workspace_root(args.workspace)
    print(f"{args.command}: configuration {path.name}, workspace {workspace}")
    if args.command == "build":
        try:
            workflow.build(workspace)
        except _input_errors() as problem:
            print(_input_problem(problem, path), file=sys.stderr, flush=True)
            raise SystemExit(1) from None
    elif args.command == "solve":
        workflow.solve(
            workspace,
            inputs=args.inputs,
            run_name=args.run_name,
            p=args.p,
            models=tuple(args.models.split(",")),
            district_warm_start=args.district_warm_start,
            confirm_large=args.yes,
        )
    elif args.against_inputs:
        run = workspace / "runs" / args.run_name if args.run_name else None
        workflow.compare_classified(
            workspace,
            run=run,
            other_inputs=Path(args.against_inputs),
            other_run=Path(args.against_run) if args.against_run else None,
        )
    elif args.inputs_only:
        report = workflow.compare_inputs(workspace, reference_inputs=args.reference_inputs)
        raise SystemExit(0 if report["consistent"] else 1)
    else:
        run = workspace / "runs" / args.run_name if args.run_name else None
        report = workflow.compare(
            workspace,
            run=run,
            reference_run=args.reference_run,
            reference_inputs=args.reference_inputs,
        )
        raise SystemExit(0 if report["consistent"] else 1)


INPUT_PROBLEM = "INPUT PROBLEM:"  # guided_run shows the lines from here on when a build stops


def _input_errors() -> tuple[type[Exception], ...]:
    """The errors that mean the input data or settings need correcting."""
    from .candidate_sources import CandidateError
    from .config import ConfigError
    from .demand_modes import DemandError
    from .schema import SchemaError
    from .validation import ValidationError
    from .vulnerability_modes import VulnerabilityError

    return (CandidateError, ConfigError, DemandError, SchemaError, ValidationError, VulnerabilityError)


def main() -> None:
    """Run the selected sub-command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "check":
        from .config import check_city

        ok, lines = check_city(_config_path(args))
        print("\n".join(lines))
        print("READY" if ok else "NOT READY: fix the problems above")
        raise SystemExit(0 if ok else 1)

    if args.command in ("build", "solve", "compare"):
        _workflow(args)
        return

    parser.error(f"Unknown command: {args.command}")
