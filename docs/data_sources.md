# Data sources

Raw data are not stored in git. Every team member downloads them locally (or in Colab) into
`data/raw/` with the commands below. All sources are free and openly licensed; please keep the
attribution notes when publishing figures.

## 1. Historical accidents — STATS19

* **Provider:** Department for Transport (DfT), Road Safety Data
* **Page:** https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
* **Licence:** Open Government Licence v3.0
* **Files:** one collision CSV per year,
  `https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-<year>.csv`
* **Download:** `python scripts/download_data.py stats19 --years 2018 2019 2020 2021 2022 2023 2024`
* **Key columns:** `collision_index`, `date`, `time`, `latitude`, `longitude`, `police_force`,
  `legacy_collision_severity`, `road_type`, `speed_limit`, `junction_detail`.
* **London filter:** `police_force` 1 (Metropolitan Police) and 48 (City of London Police).
* **Notes:**
  * Values are coded as numbers; the *Open dataset data guide* (Excel file on the same page)
    decodes them.
  * Older files use `accident_*` column names; `uarps.data.stats19` renames them.
  * 2024 is the first year under the new STATS19 specification; some scene variables changed.
  * `light_conditions`, `weather_conditions` and `road_surface_conditions` are recorded after a
    collision and are used **only for EDA**, never as model inputs.

## 2. Traffic conditions — DfT road traffic statistics

* **AADF by count point:** https://roadtraffic.dft.gov.uk/downloads → "Data at the count point
  level" → AADF file. Save as `data/raw/traffic/aadf.csv`.
* **Time-of-day profile:** table TRA0307 (traffic distribution by time of day and day of the
  week), https://www.gov.uk/government/statistical-data-sets/road-traffic-estimates-tra.
  Save as `data/raw/traffic/tra0307.ods`.
* **Licence:** Open Government Licence v3.0
* **Notes:** estimates for minor roads are less robust than for major roads. The
  `estimation_method` column shows whether a value was counted or estimated.

## 3. Weather — Open-Meteo Historical Weather API

* **Docs:** https://open-meteo.com/en/docs/historical-weather-api
* **Licence:** CC BY 4.0 (attribution: "Weather data by Open-Meteo.com")
* **Download:** `python scripts/download_data.py weather --start 2018-01-01 --end 2024-12-31`
  (about 60 grid points at 0.1° spacing; one request per point).
* **Variables:** temperature, relative humidity, precipitation, rain, snowfall, cloud cover,
  wind speed and gusts (see `configs/config.yaml`).

## 4. Road characteristics — OpenStreetMap

* **Access:** OSMnx (`uarps.data.osm`)
* **Licence:** Open Database Licence (ODbL), "© OpenStreetMap contributors"
* **Notes:** the network reflects its current state, not each study year (see limitations in
  the paper).

## 5. Calendar and boundaries

* **Bank holidays:** Python `holidays` package (`holidays.UnitedKingdom(subdiv="ENG")`).
* **Greater London boundary:** London Datastore, *Statistical GIS Boundary Files for London*,
  https://data.london.gov.uk/ — save to `data/external/`.

## Folder conventions

```
data/raw/stats19/      original collision CSVs
data/raw/traffic/      AADF and TRA0307
data/raw/weather/      one CSV per grid point + points.csv
data/external/         London boundary, OSM network cache
data/interim/          cleaned per-source tables (Parquet)
data/processed/        final cell-window tables per year (Parquet)
```
