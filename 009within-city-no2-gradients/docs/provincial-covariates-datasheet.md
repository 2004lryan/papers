# Data card: Provincial covariates, 2017-2023

| Field | Value |
|---|---|
| File (place it at) | `data/建模面板-2017_2023.csv` |
| Built by | an earlier-stage preprocessing script that is not part of this release |
| Source | National Bureau of Statistics of China, provincial annual data (https://data.stats.gov.cn) |
| Licence / availability | Public release; National Bureau of Statistics terms of service. Not redistributed in this repository; available from the corresponding author on reasonable request. |
| Rows x columns | 217 x 100 |
| SHA-256 | `e2c47e5c9643c2e2ec83265a3004e11383de8b05a5b70253f8e99fc10bcb6edf` |

## Columns

Column names are kept in Chinese because the scripts read them by name.

| Column(s) | Meaning |
|---|---|
| `省级行政区 / 年份` | province / year |
| `民用汽车拥有量` | civilian vehicle stock, 10,000 vehicles (the only column the analysis reads; it is the denominator of the provincial electric share) |

## Notes

The table holds 100 provincial columns from an earlier stage of the project; code/48_run_station_panel.py and code/65_revision_robustness.py read only 民用汽车拥有量.
