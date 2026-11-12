"""The network download: incomplete, out-of-date or failed answers are never accepted. No internet is used."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import geopandas as gpd
import pytest
import requests
from shapely.geometry import box

from childcare_access import osm_download as osm

TODAY = datetime(2026, 10, 31, 12, tzinfo=UTC)
FRESH = "2026-10-31T10:00:00Z"


def _city(size=0.01):
    return gpd.GeoDataFrame({"id": [1]}, geometry=[box(7.0, 51.0, 7.0 + size, 51.0 + size)], crs=4326)


def _answer(stamp=FRESH, way=1, size=0.01):
    area = osm.study_area(_city(size))
    west, south, east, north = area.bounds
    return {
        "version": 0.6,
        "osm3s": {"timestamp_osm_base": stamp},
        "elements": [
            {"type": "node", "id": 1, "lon": west, "lat": south},
            {"type": "node", "id": 2, "lon": east, "lat": north},
            {"type": "way", "id": way, "nodes": [1, 2], "tags": {"highway": "footway"}},
        ],
    }


def _run(tmp_path, post, city=None):
    return osm.download_network(
        city if city is not None else _city(),
        tmp_path / "highways.json",
        say=lambda text: None,
        post=post,
        wait=0,
        today=TODAY,
    )


def test_the_query_keeps_every_highway_way_except_the_exclusions():
    text = osm.query((51.0, 7.0, 51.1, 7.1))
    assert (
        "[out:json]" in text
        and "[bbox:51.000000,7.000000,51.100000,7.100000]" in text
        and 'way["highway"]["highway"!~"^(abandoned|' in text
    )
    assert "motorway_link" in text and text.endswith("out body;") and ">;" in text


def test_every_request_identifies_itself(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return _answer()

    monkeypatch.setattr(osm.requests, "post", lambda url, **kw: seen.update(kw) or Response())
    osm._post(osm.SERVERS[0], "query")
    assert seen["headers"]["User-Agent"] == osm.USER_AGENT and seen["data"] == {"data": "query"}


def test_a_failed_server_is_replaced_by_the_next(tmp_path):
    used = []

    def post(url, text):
        used.append(url)
        if len(used) == 1:
            raise requests.HTTPError("502")
        return _answer()

    assert _run(tmp_path, post)["ways"] == 1 and used[:2] == list(osm.SERVERS[:2])


def test_an_out_of_date_mirror_is_refused(tmp_path):
    answers = iter([_answer("2026-06-01T08:52:28Z"), _answer()])
    figures = _run(tmp_path, lambda url, text: next(answers))
    assert figures["date"] == FRESH


def test_a_result_cut_off_by_the_server_is_refused(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {**_answer(), "remark": "runtime error: Query timed out"}

    monkeypatch.setattr(osm.requests, "post", lambda url, **kw: Response())
    with pytest.raises(ValueError, match="incomplete"):
        osm._post(osm.SERVERS[0], "query")


def test_a_tile_that_keeps_failing_is_split_into_four(tmp_path):
    tiles = []

    def post(url, text):
        if len(tiles) < osm.ROUNDS * len(osm.SERVERS):
            tiles.append(text)
            raise requests.Timeout("timeout")
        return _answer()

    figures = _run(tmp_path, post)
    assert figures["parts"] >= 4


def test_giving_up_says_what_to_do_and_finished_tiles_are_kept(tmp_path):
    def post(url, text):
        raise requests.Timeout("timeout")

    with pytest.raises(osm.DownloadFailed, match="Run this step again later"):
        _run(tmp_path, post)


def test_a_second_run_reuses_finished_tiles(tmp_path):
    calls = []
    _run(tmp_path, lambda url, text: calls.append(url) or _answer())
    first = len(calls)
    _run(tmp_path, lambda url, text: calls.append(url) or _answer())
    assert len(calls) == first


def test_a_saved_tile_that_went_out_of_date_is_fetched_again(tmp_path):
    _run(tmp_path, lambda url, text: _answer())
    for tile in (tmp_path / "highways_tiles").glob("*.json"):
        tile.write_text(json.dumps(_answer("2026-06-01T00:00:00Z")), encoding="utf-8")
    calls = []
    _run(tmp_path, lambda url, text: calls.append(url) or _answer())
    assert calls


def test_tiles_are_merged_without_repeats_and_the_newest_date_is_recorded(tmp_path):
    stamps = iter(["2026-10-30T10:00:00Z", FRESH] * 10)
    figures = _run(tmp_path, lambda url, text: _answer(next(stamps), size=0.2), _city(0.2))
    assert figures["ways"] == 1 and figures["date"] == FRESH and figures["earliest_date"] == "2026-10-30T10:00:00Z"
    assert (tmp_path / "highways.record.json").is_file()


def test_a_network_with_missing_points_is_refused():
    network = _answer()
    network["elements"] = [e for e in network["elements"] if e["id"] != 2 or e["type"] != "node"]
    with pytest.raises(osm.DownloadFailed, match="incomplete"):
        osm.check_network(network, osm.study_area(_city()))


def test_a_network_covering_part_of_the_city_is_refused():
    network = _answer()
    network["elements"][1].update(lon=network["elements"][0]["lon"] + 0.001, lat=network["elements"][0]["lat"] + 0.001)
    with pytest.raises(osm.DownloadFailed, match="covers only"):
        osm.check_network(network, osm.study_area(_city()))


def test_retry_messages_are_plain_and_a_silent_server_still_shows_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(osm, "HEARTBEAT_S", 0.01)
    answers = iter(
        [
            requests.HTTPError("502 Server Error: Bad Gateway for url: https://example"),
            _answer("2026-06-01T08:52:28Z"),
            _answer(),
        ]
    )
    said = []

    def post(url, text):
        time.sleep(0.05)
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    osm.download_network(_city(), tmp_path / "highways.json", say=said.append, post=post, wait=0, today=TODAY)
    text = " ".join(said)
    assert (
        "this server is busy" in text
        and "has out-of-date map data" in text
        and "still working" in text
        and "Nothing for you to do" in text
    )
    assert "502" not in text and "http" not in text
