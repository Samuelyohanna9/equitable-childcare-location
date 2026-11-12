# Münster data: sources

The Münster case study (`muenster_example.ipynb`, `config/muenster.toml`) uses the data below. None of the files are
part of this repository. The SHA-256 values identify the exact files used in the thesis; a newer download of the same
dataset may differ.

| Data (path in `config/muenster.toml`) | Source | Version used | Licence |
| --- | --- | --- | --- |
| Children aged 0-2 and 3-5 per district, 2026 (`data/muenster/muenster_population_forecast.csv`) | Stadt Münster, *Kleinräumige Bevölkerungsprognose 2025-2035*, Open Data Münster: <https://opendata.stadt-muenster.de/dataset/kleinr%C3%A4umige-bev%C3%B6lkerungsprognose-m%C3%BCnster> (file `KBP_2025_2035_utf8.csv`) | forecast data as of 03/2026; SHA-256 `c96d3ce855b6f07935e5e9265b31f0976cb0b715706e1365473f0edb08ca94b1` | DL-DE-BY-2.0; Datenquelle: Stadt Münster |
| Social indicators per district, 31 Dec 2025 (`data/muenster/muenster_social_indicators.csv`) | Stadt Münster, *Bevölkerungsindikatoren - Soziales*, Open Data Münster: <https://opendata.stadt-muenster.de/dataset/bev%C3%B6lkerungsindikatoren-soziales> (file `05515000_csv_indikatoren_sozial.csv`) | 31 Dec 2014-2025; SHA-256 `15543c4a7d6243d5b0266f83b5bcab86352e0998c07d909daa84055c281f4706` | DL-DE-BY-2.0 as stated on the city portal (the openNRW copy of the record lists another licence); Datenquelle: Stadt Münster |
| 45 statistical districts (`data/muenster/muenster_districts.geojson`) | Stadt Münster, *Stadtteile (Statistische Bezirke) - Shape*, Open Data Münster: <https://opendata.stadt-muenster.de/node/40858>, reprojected to WGS84 | as of 31 Dec 2021 | DL-DE-BY-2.0; Datenquelle: Stadt Münster |
| Building footprints (`data/muenster/alkis/gru_vereinf_05515000_Muenster_EPSG25832.gpkg`) | Geobasis NRW, ALKIS *Grundrissdaten vereinfacht* (GeoPackage), Münster: <https://www.opengeodata.nrw.de/produkte/geobasis/lk/akt/gru_vereinfacht_gpkg/> (the GeoPackage inside the zip, file name changed to ASCII) | package of 29 July 2026; SHA-256 `571d10cc6f60218d3202d62e94eb8fe1a95e8b2f24cff70c91c74575c0909609` | DL-DE-Zero-2.0 |
| Children under 3 and 3-5 per 100 m cell (`data/muenster/census2022/Alter_in_infrastrukturellen_Altersgruppen.zip`) | Statistische Ämter des Bundes und der Länder, Zensus 2022: <https://www.destatis.de/static/DE/zensus/gitterdaten/Alter_in_infrastrukturellen_Altersgruppen.zip> | reference date 15 May 2022; SHA-256 `b6ffd5604ec18eed5596f1a3b83b9fd333d79019ef53cf5d1910e1cbce330758` | DL-DE-BY-2.0 |
| Walking network (`data/muenster/osm/overpass_highways_2026-06-20_merged.json`) | OpenStreetMap via the Overpass API, historical query `[date:"2026-06-20T19:49:26Z"]`: every `highway=*` way except abandoned, bus_guideway, construction, disused, motorway(_link), no, planned, proposed, raceway, razed, rest_area and services | snapshot 2026-06-20T19:49:26Z | © OpenStreetMap contributors, ODbL 1.0 |
| Existing Kitas (`amenity=kindergarten`) and OSM building tags | OpenStreetMap, read from an OSMnx download cache of the same date | 2026-06-20 | © OpenStreetMap contributors, ODbL 1.0 |
| Reviewed student residences (`data/muenster/student_residence_decisions.csv`) | the author: 82 reviewed residences (41 excluded as student-only, 41 kept), with evidence links | SHA-256 `bfa00beee77938626d07d695dcf1da7d745b83dc4bb5b41aa72f71f2b57e9292` | available from the author |

Notes:

- **Exact reproduction.** The thesis results were computed from the model inputs built from these files. The OSMnx
  cache of 2026-06-20 (existing Kitas and building tags) is not distributed because it contains personal e-mail
  addresses from OSM tags. A rebuild with Kitas downloaded today will differ slightly. The final model inputs and
  solutions of the thesis are available from the author; with them, `muenster_example.ipynb` re-runs exactly.
- **Census grid.** Values are perturbed by the Cell-Key Method; "–" (exactly zero or changed to zero) is read as 0.
- **ALKIS** is updated continuously; a fresh download describes a later state.
