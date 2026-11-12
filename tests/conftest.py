"""Shared test configuration.

Marker ``gurobi``: tests that solve models and need ``gurobipy`` with a valid licence. They are skipped, with the
reason, when either is missing. Select or exclude them with ``-m gurobi`` / ``-m "not gurobi"``.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gurobi: needs gurobipy and a valid Gurobi licence")


def _gurobi_unavailable() -> str | None:
    try:
        import gurobipy as gp
    except ImportError:
        return "gurobipy is not installed (install with: uv sync)"
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        env.dispose()
    except gp.GurobiError as error:
        return f"no valid Gurobi licence ({error.errno})"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not any("gurobi" in item.keywords for item in items):
        return
    reason = _gurobi_unavailable()
    if reason:
        for item in items:
            if "gurobi" in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))
