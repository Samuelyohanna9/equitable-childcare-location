"""Machine-readable provenance of a version-2 build: ``inputs/provenance.json``.

What was built, from what, and how: city and configuration, input files (paths and SHA-256), the demand,
vulnerability and candidate methods with their assumptions and results (priority districts, excluded sites), the
walking network, CRS, method parameters and software versions. Three identities are kept apart:

- ``scientific``: exact canonical content of every standardised solver input (``comparison.fingerprint_inputs``);
- ``descriptive_labels``: the labels (demand description, network name, ...) - a label change alone is not a
  scientific change;
- ``file_bytes``: SHA-256 of the saved files - pickles and GeoPackages are not byte-reproducible between builds,
  so different bytes with identical content are expected.

The record has no time stamps (it is deterministic for the same inputs; the build report has the run time).
Secrets never enter it: values under secret-like keys and strings that look like credentials or licence lines are
replaced by ``<redacted>`` and the redacted paths are listed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd

    from .config import CityConfig

SCHEMA = "childcare-access provenance 1"
SECRET_WORDS = {
    "password",
    "passwd",
    "pass",
    "secret",
    "token",
    "credential",
    "credentials",
    "auth",
    "authorization",
}
SECRET_JOINED = (
    "apikey",
    "licenseid",
    "licenceid",
    "privatekey",
    "secretkey",
    "accesstoken",
    "wlsaccessid",
    "wlssecret",
)
SECRET_VALUE = re.compile(
    r"licenseid|license id|wlsaccessid|wlssecret|academic license|-----begin [a-z ]*private key|"
    r"ghp_[0-9a-z]{20,}|akia[0-9a-z]{16}|(api[_-]?key|password|token)\s*[:=]",
    re.I,
)


def secret_key(name: str) -> bool:
    """Whether a key names a secret.

    A whole word (``token``, ``password``, ...) or a joined form (``api_key``, ``license_id``) counts; parts of other
    words (``rawls_stage2_tolerances``) do not.
    """
    name = str(name).lower()
    return bool(set(re.split(r"[^a-z0-9]+", name)) & SECRET_WORDS) or any(
        j in re.sub(r"[^a-z0-9]", "", name) for j in SECRET_JOINED
    )


def redact(value: object, path: str = "", found: list | None = None) -> object:
    """``value`` with secrets replaced by ``<redacted>``; the dotted paths of all replacements go to ``found``."""
    found = [] if found is None else found
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            where = f"{path}.{key}" if path else str(key)
            if secret_key(key) and item not in ("", None):
                out[key] = "<redacted>"
                found.append(where)
            else:
                out[key] = redact(item, where, found)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item, f"{path}[{i}]", found) for i, item in enumerate(value)]
    if isinstance(value, str) and SECRET_VALUE.search(value):
        found.append(path)
        return "<redacted>"
    return value


def build_record(
    config: CityConfig,
    raw: dict,
    priority: dict,
    candidates: dict,
    origins: pd.DataFrame,
    graph: nx.Graph,
    rule_set: str,
    environment: dict,
    description: str,
    folder: Path,
) -> dict:
    """The provenance of the inputs in ``folder`` (called by ``workflow.build`` after saving)."""
    from .comparison import fingerprint_inputs

    fingerprint = fingerprint_inputs(folder)
    methods = config.methods
    record = {
        "schema": SCHEMA,
        "city": {"name": config.get("city.name"), "scenario": config.get("city.scenario")},
        "configuration": {
            "file": config.source.name,
            "schema_version": config.version,
            "settings": config.data,
        },
        "inputs": raw,
        "demand": {
            **methods["demand"],
            "description": description,
            "origin_types": origins["origin_type"].value_counts().sort_index().to_dict(),
            "age_split": all(c in origins for c in ("age_0_2", "age_3_5")),
        },
        "vulnerability": priority,
        "candidates": candidates,
        "facilities": methods["facilities"],
        "network": {
            "snapshot": config.get("osm.network_snapshot"),
            "snapshot_timestamp": config.get("osm.snapshot_timestamp"),
            "rule_set": rule_set,
            "base_graph": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
        },
        "crs": {
            "projected": config.get("crs.projected"),
            **({"census": config.get("demand.census.crs")} if methods["demand"]["mode"] == "census_hybrid" else {}),
        },
        "parameters": config.parameters,
        "software": environment,
        "identity": {
            "scientific": {k: v for k, v in fingerprint["exact"].items() if k != "labels"},
            "descriptive_labels": fingerprint["exact"]["labels"],
            "file_bytes": fingerprint["file_sha256"],
            "numerical_summary": fingerprint["summary"],
        },
    }
    found: list = []
    record = redact(record, found=found)
    record["redacted"] = found
    return record


def write(folder: Path, record: dict) -> Path:
    """Write ``provenance.json`` into ``folder``; return its path."""
    path = Path(folder) / "provenance.json"
    path.write_text(json.dumps(record, indent=1, default=str, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load(folder: str | Path) -> dict | None:
    """The provenance of an inputs folder, or None (version-1 builds and the saved Münster inputs have none)."""
    path = Path(folder) / "provenance.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    missing = [
        k
        for k in (
            "schema",
            "city",
            "configuration",
            "demand",
            "vulnerability",
            "candidates",
            "network",
            "identity",
        )
        if k not in record
    ]
    if missing:
        record["incomplete"] = missing
    return record
