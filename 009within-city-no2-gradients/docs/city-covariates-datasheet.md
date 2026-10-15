# Data card: City covariates, 2017-2023

| Field | Value |
|---|---|
| File (place it at) | `data/国家统计局-城市级协变量-2017_2023.csv` |
| Built by | code/49_fetch_city_covariates.py |
| Source | National Bureau of Statistics of China, main-city annual data (https://data.stats.gov.cn) |
| Licence / availability | Public release; National Bureau of Statistics terms of service. Not redistributed in this repository; available from the corresponding author on reasonable request. |
| Rows x columns | 252 x 12 |
| SHA-256 | `c48c50578c424e0a6965b1d7c466095b4e09eef82324f550bd24a6bdb7ed610f` |

## Columns

Column names are kept in Chinese because the scripts read them by name.

| Column(s) | Meaning |
|---|---|
| `城市 / 城市代码 / 年份` | city / city code / year |
| `城市地区生产总值 / 城市第二产业增加值 / 城市第三产业增加值` | GDP / value added of the secondary / tertiary sector |
| `城市年末户籍人口` | registered population at year end |
| `城市旅客运输量 / 城市货物运输量` | passenger / freight transport volume |
| `城市社会消费品零售总额` | retail sales of consumer goods |
| `环境噪声等效声级 / 道路交通等效声级` | ambient / road-traffic equivalent noise level |

## Notes

Covers the 36 main cities in the bureau's database, a subset of the 307 cities.
