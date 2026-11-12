"""The guided run: P defaults to 3 and is checked, secrets stay out of output and files, and clear messages."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from childcare_access import guided_run as guide

SECRET = "a1b2c3d4-secret-value"


def _wls():
    return guide.Licence(
        "wls",
        fingerprint=guide._fingerprint([SECRET]),
        _values={"WLSACCESSID": "access-77", "WLSSECRET": SECRET, "LICENSEID": "998877"},
    )


def _city(tmp_path):
    return guide.City("Test", tmp_path / "c.toml", tmp_path, tmp_path / "inputs", "")


def test_p_defaults_to_three(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert guide.DEFAULT_P == 3 and guide.choose_p() == 3


@pytest.mark.parametrize("answer", [5, "5", " 5 "])
def test_p_can_be_chosen(answer):
    assert guide.choose_p(answer) == 5


@pytest.mark.parametrize("answer", ["0", "-2", "2.5", "three", ""])
def test_an_invalid_p_is_refused_with_a_message(answer):
    with pytest.raises(guide.GuidedRunError, match="whole number of at least 1"):
        guide.choose_p(answer)


@pytest.mark.parametrize(
    "line",
    [
        "Academic license 1234567 - for non-commercial use only",
        "Set parameter LicenseID to value 998877",
        "Set parameter WLSAccessID",
        "registered to someone@example.org",
        "Using license file C:/x/gurobi.lic",
    ],
)
def test_licence_lines_are_hidden(line):
    assert guide.redact(line) == "[licence line hidden]"


def test_secrets_and_email_addresses_are_removed():
    assert SECRET not in guide.redact(f"value {SECRET} here", [SECRET])
    assert "someone@example.org" not in guide.redact("contact someone@example.org")


def test_ordinary_solver_lines_are_kept():
    line = "  1234  56  5.7158 0.0% 12s"
    assert guide.redact(line) == line


def test_the_licence_never_shows_its_secret():
    licence = _wls()
    assert SECRET not in repr(licence) and SECRET not in str(licence) and SECRET not in licence.fingerprint


def test_the_fingerprint_is_salted():
    assert guide._fingerprint([SECRET]) != guide._fingerprint([SECRET])


def test_the_temporary_licence_file_is_deleted_also_after_an_error(tmp_path):
    with pytest.raises(RuntimeError), guide._licence_environment(_city(tmp_path), _wls()) as environment:
        file = Path(environment["GRB_LICENSE_FILE"])
        assert file.is_file() and SECRET in file.read_text(encoding="utf-8")
        raise RuntimeError("solver failed")
    assert not file.exists()


def test_the_live_report_hides_secrets_and_reports_a_failure(tmp_path, capsys):
    code = f"print('Set parameter LicenseID to value 1'); print('key {SECRET}'); print('working'); raise SystemExit(3)"
    with pytest.raises(guide.GuidedRunError) as error:
        guide.run_live("Test", [sys.executable, "-c", code], dict(os.environ), secrets=[SECRET])
    out = capsys.readouterr().out
    assert SECRET not in out and SECRET not in str(error.value) and "LicenseID" not in out and "working" in out


def test_a_missing_configuration_gets_a_useful_message():
    with pytest.raises(guide.GuidedRunError, match="APPLYING_TO_ANOTHER_CITY"):
        guide.choose_city("no/such/city.toml")


def test_results_show_the_change_table_and_box_plots(tmp_path, capsys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from childcare_access.guided_run import show_results

    pd.DataFrame({"model": ["Utilitarian"]}).to_csv(tmp_path / "certification.csv", index=False)
    pd.DataFrame({"sites": ["a"]}).to_csv(tmp_path / "sites.csv", index=False)
    pd.DataFrame({"Current": [1.0]}, index=["City mean"]).to_csv(tmp_path / "indicators.csv")
    pd.DataFrame({"Utilitarian": [-2.5]}, index=["City mean"]).to_csv(tmp_path / "change_vs_current_percent.csv")
    pd.DataFrame({"Utilitarian": [3.0]}, index=["City child-min saved"]).to_csv(tmp_path / "savings.csv")
    for name in (
        "sites_map.png",
        "savings_grid_maps.png",
        "priority_group_walks.png",
        "walking_time_boxplots.png",
    ):
        plt.figure(figsize=(1, 1)).savefig(tmp_path / name)
    show_results(tmp_path)
    assert str(tmp_path) in capsys.readouterr().out


def test_a_sample_settings_file_is_made_from_the_template_and_never_replaced(tmp_path, monkeypatch):
    import tomllib

    from childcare_access.config import check_city

    (tmp_path / "config").mkdir()
    monkeypatch.setattr(guide, "PROJECT_ROOT", tmp_path)
    made = guide.sample_settings("Neu Stadt")
    data = tomllib.loads(made.read_text(encoding="utf-8"))
    assert made.name == "neu_stadt.toml" and data["city"] == {"name": "Neu Stadt", "scenario": "neu_stadt"}
    assert data["workspace"]["root"] == "workspace/neu_stadt"
    ok, lines = check_city(made)
    assert not ok and any("REPLACE" in line for line in lines)  # the check names what is still to fill in
    made.write_text("# half filled\n", encoding="utf-8")
    assert guide.sample_settings("Neu Stadt") == made and made.read_text(encoding="utf-8") == "# half filled\n"


def test_an_input_problem_ends_the_build_with_a_short_message():
    from childcare_access import cli
    from childcare_access.schema import SchemaError

    text = cli._input_problem(
        SchemaError("child_totals: 1 problem(s):\n  - district_id not in the districts: ['1']"), Path("dortmund.toml")
    )
    assert text.startswith(cli.INPUT_PROBLEM) and "choose the matching columns in dortmund.toml" in text
