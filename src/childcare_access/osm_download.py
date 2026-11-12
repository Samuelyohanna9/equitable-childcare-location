"""Download of a walking-network snapshot from Overpass, for cities that do not have one.

The query is the one used for Münster: every ``highway`` way except the values in ``EXCLUDED_HIGHWAYS``, so that the
pedestrian rules see the same kind of data as for Münster. A city is requested in tiles, because public servers often
refuse a whole city. A tile that fails is split into four. Finished tiles are kept, so an interrupted download resumes.
A result counts only if Overpass reports no problem, and the merged file is checked before it is saved. Answers with
old map data are refused, because some mirror servers lag by months.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

Tile = tuple[float, float, float, float]  # south, west, north, east (degrees)
Say = Callable[[str], None]  # where progress messages for the reader go
Post = Callable[[str, str], dict]  # (server URL, query) -> parsed Overpass answer

EXCLUDED_HIGHWAYS = (
    "abandoned|bus_guideway|construction|disused|motorway|motorway_link|no|planned|proposed|raceway|razed|"
    "rest_area|services"
)
SERVERS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
USER_AGENT = "childcare-access research notebook (non-commercial)"
BUFFER_M = 2000  # children near the edge can walk across the city boundary
TILE_DEGREES = 0.08  # about 6-9 km; 16 tiles of this size worked for Dortmund
MAX_SPLITS = 3
ROUNDS = 3  # tries of all servers before a tile is split or given up
QUERY_TIMEOUT_S = 300
HEARTBEAT_S = 30  # a server can stay silent for minutes; the reader should see that the work goes on
MAX_AGE_DAYS = 7  # some mirrors answer normally with data months old


class DownloadFailed(RuntimeError):
    """The servers did not deliver a complete network; the message says what to do."""


def study_area(districts: gpd.GeoDataFrame, buffer_m: float = BUFFER_M) -> BaseGeometry:
    """The districts with a buffer, in longitude and latitude."""
    metric = districts.to_crs(districts.estimate_utm_crs())
    return gpd.GeoSeries([metric.union_all().buffer(buffer_m)], crs=metric.crs).to_crs(4326).iloc[0]


def tiles_for(area: BaseGeometry, size: float = TILE_DEGREES) -> list[Tile]:
    """South, west, north, east of the grid tiles that touch the area."""
    west, south, east, north = area.bounds
    columns, rows = (
        max(1, math.ceil((east - west) / size)),
        max(1, math.ceil((north - south) / size)),
    )
    width, height = (east - west) / columns, (north - south) / rows
    tiles = [
        (south + j * height, west + i * width, south + (j + 1) * height, west + (i + 1) * width)
        for i in range(columns)
        for j in range(rows)
    ]
    return [t for t in tiles if area.intersects(box(t[1], t[0], t[3], t[2]))]


def query(tile: Tile) -> str:
    """Overpass query for the walkable ways of one tile."""
    south, west, north, east = tile
    return (
        f"[out:json][timeout:{QUERY_TIMEOUT_S}][bbox:{south:.6f},{west:.6f},{north:.6f},{east:.6f}];"
        f'(way["highway"]["highway"!~"^({EXCLUDED_HIGHWAYS})$"];>;);out body;'
    )


def _split(tile: Tile) -> list[Tile]:
    """The four quarters of a tile (for a tile the server cannot answer in time)."""
    south, west, north, east = tile
    middle_y, middle_x = (south + north) / 2, (west + east) / 2
    return [
        (south, west, middle_y, middle_x),
        (south, middle_x, middle_y, east),
        (middle_y, west, north, middle_x),
        (middle_y, middle_x, north, east),
    ]


def _post(url: str, text: str) -> dict:
    """One answer of a server. Raises if it is refused, unreadable or incomplete."""
    response = requests.post(url, data={"data": text}, headers={"User-Agent": USER_AGENT}, timeout=QUERY_TIMEOUT_S + 30)
    response.raise_for_status()
    result = response.json()
    if result.get("remark"):  # a server that runs out of time still answers 200, with part of the data
        raise ValueError(f"incomplete result: {result['remark'][:100]}")
    if "elements" not in result or "osm3s" not in result:
        raise ValueError("not an Overpass result")
    return result


def _is_current(result: dict, today: datetime) -> bool:
    """Whether a saved answer is recent enough to be reused."""
    stamp = datetime.fromisoformat(result["osm3s"]["timestamp_osm_base"].replace("Z", "+00:00"))
    return today - stamp <= timedelta(days=MAX_AGE_DAYS)


def _waiting(post: Post, url: str, text: str, say: Say) -> dict:
    """The server's answer, with a line now and then so that a slow server does not look like a stopped download."""
    with ThreadPoolExecutor(1) as pool:
        job, waited = pool.submit(post, url, text), 0
        while True:
            try:
                return job.result(timeout=HEARTBEAT_S)
            except TimeoutError:
                if job.done():
                    raise
                waited += HEARTBEAT_S
                say(f"still working, the server is slow ({waited} seconds so far). Nothing for you to do.")


def _fetch_tile(tile: Tile, folder: Path, say: Say, post: Post, wait: float, today: datetime) -> list[Path]:
    """The saved file of this tile. A file already saved is reused while its data are current."""
    text = query(tile)
    path = folder / f"{hashlib.sha256(text.encode()).hexdigest()[:16]}.json"
    if path.is_file() and _is_current(json.loads(path.read_text(encoding="utf-8")), today):
        return [path]
    problem = ""
    for attempt in range(ROUNDS * len(SERVERS)):
        url = SERVERS[attempt % len(SERVERS)]
        try:
            result = _waiting(post, url, text, say)
            if not _is_current(result, today):
                raise ValueError(f"out-of-date map data of {result['osm3s']['timestamp_osm_base'][:10]}")
        except (requests.RequestException, ValueError) as error:
            problem = f"{url.split('/')[2]}: {str(error)[:90]}"
            reason = "has out-of-date map data" if str(error).startswith("out-of-date") else "is busy"
            say(f"this server {reason}, so the notebook is trying another one by itself. Nothing for you to do.")
            time.sleep(wait * (attempt + 1))
            continue
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        partial.replace(path)  # never a half-written tile, if the download stops here
        time.sleep(wait)
        return [path]
    raise DownloadFailed(problem)


def _fetch(tile: Tile, folder: Path, say: Say, post: Post, wait: float, today: datetime, splits: int = 0) -> list[Path]:
    """The saved files of a tile; a tile that fails is split into quarters (up to the split limit)."""
    try:
        return _fetch_tile(tile, folder, say, post, wait, today)
    except DownloadFailed as failed:
        if splits >= MAX_SPLITS:
            raise DownloadFailed(
                f"The Overpass servers did not deliver one part of the city ({failed}). Finished parts are saved. "
                "Run this step again later to continue; the servers are often less busy at night."
            ) from None
        say(
            "this part is large for the servers right now, so the notebook is fetching it in smaller pieces. Nothing "
            "for you to do."
        )
        return [path for part in _split(tile) for path in _fetch(part, folder, say, post, wait, today, splits + 1)]


def check_network(network: dict, area: BaseGeometry) -> dict:
    """Check a merged network and return its figures. Raises if it is not usable."""
    nodes = {e["id"]: e for e in network["elements"] if e["type"] == "node"}
    ways = [e for e in network["elements"] if e["type"] == "way"]
    if not ways:
        raise DownloadFailed(
            "The result contains no streets or paths. Check that the district file is in the right place."
        )
    if any(ref not in nodes for way in ways for ref in way["nodes"]):
        raise DownloadFailed(
            "Some streets refer to points that are missing, so the network is incomplete. Run this step again."
        )
    if any("highway" not in way.get("tags", {}) for way in ways):
        raise DownloadFailed("Some ways have no highway tag, so this is not the expected network.")
    west, south, east, north = area.bounds
    lons, lats = [n["lon"] for n in nodes.values()], [n["lat"] for n in nodes.values()]
    covered = (
        (min(max(lons), east) - max(min(lons), west))
        * (min(max(lats), north) - max(min(lats), south))
        / ((east - west) * (north - south))
    )
    if covered < 0.8:
        raise DownloadFailed(
            f"The network covers only {covered:.0%} of the study area, so it is incomplete. Run this step again."
        )
    return {"ways": len(ways), "nodes": len(nodes), "covered_share": round(covered, 3)}


def download_network(
    districts: gpd.GeoDataFrame,
    target: Path,
    say: Say = print,
    post: Post = _post,
    wait: float = 8.0,
    today: datetime | None = None,
) -> dict:
    """Download, merge, check and save the network of these districts. Returns its figures and its date."""
    target, today = Path(target), today or datetime.now(UTC)
    folder = target.parent / f"{target.stem}_tiles"
    folder.mkdir(parents=True, exist_ok=True)
    area = study_area(districts)
    tiles = tiles_for(area)
    say(
        f"Downloading the walking network in {len(tiles)} parts. Public servers are often busy, so the notebook "
        "tries other servers by itself. "
        "This can take several minutes. Keep waiting until it says the network is saved."
    )
    paths = []
    for number, tile in enumerate(tiles, 1):
        paths += _fetch(
            tile,
            folder,
            lambda text, part=number: say(f"  Part {part} of {len(tiles)}: {text}"),
            post,
            wait,
            today,
        )
        say(f"  Part {number} of {len(tiles)} done")
    merged, base, dates = {}, None, set()
    for path in paths:
        part = json.loads(path.read_text(encoding="utf-8"))
        base = base or {key: value for key, value in part.items() if key != "elements"}
        dates.add(part["osm3s"]["timestamp_osm_base"])
        for element in part["elements"]:
            merged[(element["type"], element["id"])] = element
    base["osm3s"] = {**base["osm3s"], "timestamp_osm_base": max(dates)}
    network = {**base, "elements": list(merged.values())}
    figures = check_network(network, area)
    target.write_text(json.dumps(network, ensure_ascii=False), encoding="utf-8")
    figures.update(file=str(target), date=max(dates), earliest_date=min(dates), parts=len(paths))
    target.with_suffix(".record.json").write_text(
        json.dumps({**figures, "query_template": query((0, 0, 0, 0)), "buffer_m": BUFFER_M}, indent=1),
        encoding="utf-8",
    )
    say(
        f"Saved {figures['ways']:,} ways. Map data of {figures['date']}. Credit: OpenStreetMap contributors (Open "
        "Database License)."
    )
    return figures
