"""Helpers shared by the build and solve steps: pickle files, elapsed-time logging and build checks.

Example:
    >>> log = ElapsedLog()
    >>> checks = BuildChecks(log)
    >>> checks("district totals conserved", True, {"max_error": 0.0})  # doctest: +ELLIPSIS
    [    ...s] PASS district totals conserved: {'max_error': 0.0}
    >>> checks.all_passed
    True
"""

from __future__ import annotations

import pickle
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def read_pickle(path: str | Path) -> Any:
    """Load one pickled object (only for the project's own files: unpickling runs code)."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


class ElapsedLog:
    """Print messages prefixed with the seconds since this object was created, e.g. ``[   12.3s] message``."""

    def __init__(self, width: int = 8) -> None:
        self.started = time.perf_counter()
        self.width = width

    @property
    def seconds(self) -> float:
        """Seconds since this object was created."""
        return time.perf_counter() - self.started

    def __call__(self, message: str) -> None:
        """Print ``message`` with the elapsed time."""
        print(f"[{self.seconds:{self.width}.1f}s] {message}", flush=True)


class BuildChecks:
    """The checks of one build: each is logged as PASS or FAIL and kept for its validation report.

    Args:
        log: Where each check is reported.
        stop_on_failure: Exit the script at the first failed check.
        detail_chars: How much of the detail is logged (None: all of it); the report always keeps all of it.
    """

    def __init__(
        self, log: Callable[[str], None], *, stop_on_failure: bool = False, detail_chars: int | None = 300
    ) -> None:
        self.log = log
        self.stop_on_failure = stop_on_failure
        self.detail_chars = detail_chars
        self.records: list[dict[str, Any]] = []

    def __call__(self, name: str, passed: object, detail: object) -> None:
        """Record and log one check; ``detail`` is kept in the report and shown in the log."""
        self.records.append({"check": name, "passed": bool(passed), "detail": detail})
        shown = str(detail) if self.detail_chars is None else str(detail)[: self.detail_chars]
        self.log(f"{'PASS' if passed else 'FAIL'} {name}: {shown}")
        if self.stop_on_failure and not passed:
            sys.exit(f"check failed: {name}")

    @property
    def all_passed(self) -> bool:
        """Whether every check so far passed (True if there are none)."""
        return all(record["passed"] for record in self.records)
