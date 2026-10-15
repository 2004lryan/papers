# Within-city nitrogen dioxide exposure gradients in China narrowed ahead of electric vehicle fleet growth

Analysis code for the manuscript:

> **Within-city nitrogen dioxide exposure gradients in China narrowed ahead of electric vehicle fleet growth**
> Panlin Li, Hua Huang, Yanping Song\*
> Xinjiang Agricultural University · *manuscript in preparation*

Repository: https://github.com/2004lryan/papers/tree/main/009within-city-no2-gradients

---

## What the code does

National monitoring stations in China report hourly nitrogen dioxide (NO₂), but air quality is summarised as a city
average. The scripts here measure how NO₂ differs *within* a city, along the built-up fraction within 500 m of each
station, and ask whether vehicle electrification narrowed that contrast.

Using 1311 stations in 307 cities over 2015 to 2024, the analysis finds that, within a city and year, stations at the
ninetieth percentile of built-up exposure recorded 16.3% higher annual mean NO₂ than those at the tenth. The NO₂
gradient narrowed by 42.5% over the decade, fastest over 2017 to 2020, and the sulphur dioxide (SO₂) gradient narrowed
faster, by 84.6%. By 2020 the NO₂ gradient had completed 77% (95% interval 47% to 127%) of its 2017 to 2023
contraction and the national electric fleet 18% of its growth. In fixed-effects regressions the fleet–exposure
interaction was no longer detected once an exposure-specific trend was admitted.

## Repository structure

```
├── code/                  # the scripts below, numbered roughly in analysis order, and fetch_datasets.py
├── data/DATA.md           # sources, licences and SHA-256 of every table the analysis reads (no data files)
├── docs/                  # one data card per table
├── requirements.txt
├── ruff.toml              # lint settings (ruff check code)
├── CITATION.cff
└── LICENSE
```

## Scripts

`code/01_export_utils.py` holds the shared paths (`data/`, `outputs/`, `logs/` under the repository root) and the
workbook, figure and log helpers that every script loads.

| Step | Script | What it does |
|---|---|---|
| Acquisition | `42_fetch_station_air_quality.py` | Hourly station records from the public archive of the national releases |
| | `43_fetch_station_metadata.py` | Station register with coordinates and city attribution |
| | `44_station_traffic_exposure.py` | Road-network exposure from OpenStreetMap (a pilot that stopped at 12 stations) |
| | `45_station_builtup_exposure.py` | Built-up exposure at 500 m and 1000 m from the GHSL built-up surface |
| | `29_fetch_provincial_meteorology.py` | City meteorology from the NASA POWER daily point service |
| | `49_fetch_city_covariates.py` | City covariates from the National Bureau of Statistics |
| | `32_integrate_nev_stock.py`, `40_transcribe_2023_city_nev.py`, `37_build_city_panel.py` | Provincial and city new energy vehicle stock and the city panel |
| Panel | `46_build_station_panel.py`, `50_add_o3_annual_mean.py` | Station annual panel, 2015 to 2024, with O₃ and Ox annual means |
| Analysis | `47_station_pretest.py` | Pre-tests of the station design |
| | `48_run_station_panel.py` | Electrification regression (equation 3) and the robustness checks of Supplementary Table S2 |
| | `51_station_timing_and_power.py` | Annual gradient path, timing against electrification, detectability |
| | `52_station_figures.py` | Figure 2 and Supplementary Figures S1 and S2 |
| | `64_manuscript_added_figures.py` | Figures 1, 3 and 4 and the graphical abstract |
| Checks | `54_cluster_size_summary.py` | City cluster sizes in the regression sample |
| | `55_trend_collinearity_check.py` | Collinearity of the exposure-specific trend specification |
| | `56_o3_mda8_clock_check.py` | O₃ daily maximum 8-hour mean on clock windows |
| | `57_duplicate_hours_check.py`, `58_dedup_rebuild_check.py` | Repeated hourly records and a rebuild without them |
| | `59_threshold_per_pollutant_check.py` | Completeness threshold applied per pollutant |
| | `60_priority_region_city_check.py` | Blue Sky Protection Campaign priority regions defined by city list |
| | `61_gradient_break_test.py` | Segment rates of the gradient path |
| | `62_gradient_meteorology_check.py` | Gradient allowed to vary with meteorology |
| | `63_gradient_balanced_panel_check.py` | Fixed station set |
| | `65_revision_robustness.py` | Further robustness checks, including alternative fleet measures |
| | `66_exposure_epoch_check.py` | Exposure from the 2015 and interpolated GHSL epochs |
| | `67_gradient_differential_check.py` | NO₂ minus comparison-species gradient paths |
| | `68_vehicle_stock_match_check.py` | Station-years without a city vehicle stock, and columns 1 and 3 of Table 2 with the unmatched cities restored |

Most scripts write their results to a workbook in `outputs/` and a log in `logs/`, both created on first run; the
acquisition scripts also write the cleaned tables under `data/`, and `50_add_o3_annual_mean.py` adds its columns to the
station panel in place. Column names, sheet names and log messages are in Chinese because the source tables use
Chinese field names; the data cards in `docs/` translate every column the analysis uses. `write_excel_workbook`
prefixes each sheet name with its position (`01_`, `02_`, …), so sheets whose titles already carry a number appear as
`01_01_…`.

## Reproducing

```bash
pip install -r requirements.txt          # Python 3.13
python code/fetch_datasets.py --list     # paths and checksums of the tables the analysis reads
python code/fetch_datasets.py --verify   # after placing the tables under data/ and outputs/
```

Run every command from the repository root. The scripts find `data/`, `outputs/` and `logs/` from their own location,
but the download caches of scripts 42, 45 and 66 (under `data/raw/`, not tracked) are relative to the working
directory.

**From the cleaned tables.** With the eight tables of `data/DATA.md` in place, run scripts 47, 48, 51, 52, 54, 55 and
59 to 68 in numerical order, except that 65 must run before 64. Later scripts read the workbooks of earlier ones: 59
and 60 read that of 51, 64 reads those of 51, 61, 62, 63 and 65, 66 reads that of 63 and downloads the GHSL 2015
tiles on first use, and 68 reads that of 48. Script 62 reads the city meteorology workbook of script 29, one of the eight tables.

**Scripts that need the hourly files.** Scripts 56, 57 and 58, and pre-test P3 of script 47, read the hourly station
files, which are not redistributed; without them 47 skips P3 and the other three stop. Rebuild the files with

```bash
python code/42_fetch_station_air_quality.py --start 20150101 --end 20241231
```

which downloads one file for each day of 2015 to 2024, deletes each once parsed, and keeps one parquet file per year
(about 1.5 GB in total). Downloads can fail under rate limiting. The workbook `outputs/42_站点数据覆盖.xlsx` counts
the failed days of each year fetched in a run; rerunning the same command with `--only-missing` fetches only the days
absent from each year's file, and the log states for each year how many days are held and how many remain to fetch
(days the archive itself lacks stay absent). The results of the scripts that read the hourly files depend on their
completeness, and script 58 stops when the station-years it rebuilds no longer reproduce the station panel. Script 56
also reads the workbook of 48, and 58 reads those of 48, 55 and 57, so 57 runs before 58.

**Rebuilding the cleaned tables.** The acquisition scripts rebuild the tables from the public sources in this order:
42, 43, 44, 45, 29, 49, 32, 40, 37, 46 and 50. Script 44 is the road pilot, run on the first 12 stations
(`--limit 12`); it supplies only the column `加权路网密度_500m` of the station panel, and without it the panel has 57
columns and its checksum differs from `data/DATA.md`. Some sources, such as the station register and OpenStreetMap,
are live, so a rebuild will not in general reproduce the published checksums. Four inputs come from earlier-stage
preprocessing that is not included here, and 32, 37 and 40 read the vehicle-stock package placed under
`data/raw/公安部口径-新能源汽车保有量-2017_2023/` (the docstring of script 32 lists the package files the scripts
use). The package is a third-party compilation of year-end new energy vehicle registrations on the Ministry of Public
Security basis, downloaded on 2 September 2026 from a public source under the folder name
`【2017-2023】新能源汽车保有量数据(各省市)`; no stable address is recorded for that download, and script 32 checks the
package against the national totals the ministry publishes. It matches the compilation sold by Xingyuan Data
(https://www.stardata360.com, archives/2190 and archives/4529) in the yearly table totals for 2017 to 2022, rows not
allocated to a city included, and in the 2023 values for Shanghai and Zhejiang listed there; the vendor names no
source. The stock tables built from the package are available on request with the other cleaned tables, subject to
the terms of the compilation, and `data/DATA.md` lists all of these inputs. Scripts 29, 42 to 45 and 66 call the
system `curl`, and 49 drives a locally installed Chrome, Chromium or Edge through Playwright.

## Data availability

This repository contains no data. The cleaned tables are built from public sources (national monitoring releases,
GHSL R2023A under CC BY 4.0, NASA POWER, the National Bureau of Statistics, and a third-party compilation of vehicle
registration counts on the Ministry of Public Security basis) and are available from the corresponding author on
reasonable request; the two vehicle stock tables are shared subject to the terms of the compilation they derive from.
The hourly files are a reformatted subset of the public hourly archive and are not redistributed. `data/DATA.md` gives
the source, licence and SHA-256 of every table.

## Citation

See `CITATION.cff`. A BibTeX entry will be added on publication. The manuscript is not included in this repository.

## Licence

Code: MIT (see `LICENSE`). Each data source is used under its own terms (see `data/DATA.md`).

## Contact

Yanping Song, syp@xjau.edu.cn, Xinjiang Agricultural University.
