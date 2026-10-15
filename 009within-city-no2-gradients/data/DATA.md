# Datasets: sources, licences and checksums

This repository contains **no data files**. The analysis reads the eight cleaned tables below. They are built from
public sources, by the scripts in `code/` except the provincial covariate table, which comes from earlier-stage
preprocessing (see the last section). They are not redistributed here and are available from the corresponding
author on reasonable request. Place each file at the path shown; `python code/fetch_datasets.py --verify` checks
the SHA-256 of every file it finds. The 16-character prefixes are those printed in Table 1 of the manuscript.

## Cleaned tables read by the analysis

| Table | Path in this repository | Built by | Rows x columns | SHA-256 (first 16) | Data card |
|---|---|---|---|---|---|
| Station annual panel, 2015-2024 | `data/站点年面板-2015_2024.csv` | code/46_build_station_panel.py, then code/50_add_o3_annual_mean.py | 11220 x 58 | `732a01f4a452bb55` | [`docs/station-annual-panel-datasheet.md`](../docs/station-annual-panel-datasheet.md) |
| Station register with coordinates | `data/国控站点元数据.csv` | code/43_fetch_station_metadata.py | 1601 x 7 | `29964754dfafcac0` | [`docs/station-register-datasheet.md`](../docs/station-register-datasheet.md) |
| Built-up exposure at 500 m and 1000 m | `data/站点建成区暴露.csv` | code/45_station_builtup_exposure.py | 1601 x 11 | `3acbc9f46a30fe6c` | [`docs/builtup-exposure-datasheet.md`](../docs/builtup-exposure-datasheet.md) |
| Provincial new energy vehicle stock, 2017-2023 | `data/省级新能源汽车保有量-公安部口径-2017_2023.csv` | code/32_integrate_nev_stock.py | 217 x 5 | `36f46632fc39ec64` | [`docs/provincial-nev-stock-datasheet.md`](../docs/provincial-nev-stock-datasheet.md) |
| City panel with vehicle stock, 2017-2023 | `data/城市面板-2017_2023.csv` | code/37_build_city_panel.py (2023 values from code/40_transcribe_2023_city_nev.py) | 2333 x 30 | `10876f2b75a9d404` | [`docs/city-vehicle-stock-datasheet.md`](../docs/city-vehicle-stock-datasheet.md) |
| City covariates, 2017-2023 | `data/国家统计局-城市级协变量-2017_2023.csv` | code/49_fetch_city_covariates.py | 252 x 12 | `c48c50578c424e0a` | [`docs/city-covariates-datasheet.md`](../docs/city-covariates-datasheet.md) |
| Provincial covariates, 2017-2023 | `data/建模面板-2017_2023.csv` | an earlier-stage preprocessing script that is not part of this release | 217 x 100 | `e2c47e5c9643c2e2` | [`docs/provincial-covariates-datasheet.md`](../docs/provincial-covariates-datasheet.md) |
| City meteorology, 2015-2024 (workbook) | `outputs/29_气象构建过程.xlsx` (sheet `02_城市年值`) | code/29_fetch_provincial_meteorology.py | 3370 x 9 | `cadf7cd91a6f2d84` | [`docs/city-meteorology-datasheet.md`](../docs/city-meteorology-datasheet.md) |

## Public sources

| Source | Used for | Licence / terms |
|---|---|---|
| CNEMC hourly releases, archived at https://quotsoft.net/air | hourly station records (`code/42_fetch_station_air_quality.py`) | public release; the archive page welcomes analysis and research and names no licence |
| CNEMC real-time publication platform, https://air.cnemc.cn:18007/ | station register with coordinates (`code/43_fetch_station_metadata.py`) | public release |
| GHSL built-up surface R2023A, https://human-settlement.emergency.copernicus.eu | built-up exposure (`code/45_station_builtup_exposure.py`) | CC BY 4.0 |
| OpenStreetMap via the Overpass API | road-network exposure pilot, 12 stations (`code/44_station_traffic_exposure.py`) | ODbL |
| NASA POWER daily point service, https://power.larc.nasa.gov | city meteorology (`code/29_fetch_provincial_meteorology.py`) | public release |
| Open-Meteo Geocoding API, https://geocoding-api.open-meteo.com (location database from GeoNames) | city coordinates for the meteorology retrieval (`code/29_fetch_provincial_meteorology.py`) | API data under CC BY 4.0; GeoNames under CC BY |
| National Bureau of Statistics, https://data.stats.gov.cn | city covariates (`code/49_fetch_city_covariates.py`); provincial covariates (earlier-stage preprocessing, not part of this release) | public release, bureau terms of service |
| Ministry of Public Security annual bulletins, https://www.mps.gov.cn | national new energy vehicle totals | public release |
| Third-party compiled package on the ministry's registration basis, downloaded on 2 September 2026 from a public source under the folder name `【2017-2023】新能源汽车保有量数据(各省市)`; no stable address is recorded for that download. The package matches the compilation sold by Xingyuan Data (https://www.stardata360.com/archives/2190 and https://www.stardata360.com/archives/4529) in the yearly table totals for 2017 to 2022, rows not allocated to a city included, and in the 2023 values for Shanghai and Zhejiang listed there; the vendor names no source | provincial and city vehicle stock (`code/32`, `code/37`, `code/40`) | third-party compilation; not redistributed here, the stock tables built from it are available on request subject to the terms of the compilation |

## Intermediate files the scripts write but this repository does not ship

- `data/站点小时空气质量-<year>.parquet`: hourly station records reformatted from the archive by `code/42_fetch_station_air_quality.py`.
  They are a reformatted subset of the public hourly archive, not a new dataset, and are not redistributed.
- `data/站点交通暴露.csv`: road-network exposure pilot from `code/44_station_traffic_exposure.py` (12 stations).
- `data/城市坐标-地级及以上.csv`: geocoded city coordinates written by `code/29_fetch_provincial_meteorology.py`.

## Inputs built by earlier-stage preprocessing that is not part of this release

- `data/城市-省份对照表.csv`: city-to-province lookup, read by `code/29`.
- `outputs/26_省级空气质量构建过程.xlsx`, sheet `03_城市年评价值`: city annual air-quality values, read by `code/37`.
- `data/省级研究面板-2015_2024-v2.csv`: provincial research panel that `code/32` extends with the vehicle stock.
- `data/建模面板-2017_2023.csv`: the provincial covariate table listed above.

These are also available from the corresponding author on reasonable request.

Each source is used under its own terms; please cite the original providers, and for GHSL the reference
publication of release R2023A.
