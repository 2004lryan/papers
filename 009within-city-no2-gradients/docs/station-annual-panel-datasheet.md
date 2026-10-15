# Data card: Station annual panel, 2015-2024

| Field | Value |
|---|---|
| File (place it at) | `data/站点年面板-2015_2024.csv` |
| Built by | code/46_build_station_panel.py, then code/50_add_o3_annual_mean.py |
| Source | China National Environmental Monitoring Centre (CNEMC) hourly releases, retrieved from the public archive at https://quotsoft.net/air |
| Licence / availability | Public release; the archive page welcomes analysis and research and names no licence. Not redistributed in this repository; available from the corresponding author on reasonable request. |
| Rows x columns | 11220 x 58 |
| SHA-256 | `732a01f4a452bb55e613ef24550c280b1243deed24b67f9ab4dc54bb87ce3f36` |

## Columns

Column names are kept in Chinese because the scripts read them by name.

| Column(s) | Meaning |
|---|---|
| `站点编号 / 站点名称` | station code / station name |
| `年份` | year |
| `所属城市 / 城市代码 / 城市名 / 省级行政区` | city as listed in the register / city code / standardised city name / province |
| `纬度 / 经度` | latitude / longitude |
| `{NO2, PM2.5, PM10, SO2, CO}_年均` | annual mean concentration (µg m-3; CO in mg m-3) |
| `{pollutant}_有效日` | number of valid days for that pollutant (a day is valid with at least 20 valid hours) |
| `O3_8h_90分位 / O3_有效日` | 90th percentile of the daily maximum 8-hour O3 / its valid days |
| `{pollutant}_工作日 / _周末 / _周末效应` | weekday mean / weekend mean / weekend effect |
| `{pollutant}_高峰均值 / _夜间均值 / _峰谷比` | commuting-peak mean / overnight mean / peak-to-trough ratio |
| `当年数据天数` | number of dates present in that year's hourly files (the same value for every station) |
| `建成区占比_500m / 建成区占比_1000m / 建成区占比_城内百分位` | built-up fraction within 500 m / within 1000 m / within-city percentile rank of the 500 m fraction |
| `加权路网密度_500m` | weighted road density within 500 m (road-network pilot of 12 stations, 9 of them in this panel) |
| `NEV保有量_辆 / ln_NEV` | city new energy vehicle (NEV) stock in vehicles / its natural logarithm |
| `本省当年未分配占比` | share of the provincial NEV stock not allocated to any city in that year |
| `O3_年均 / O3_年均_有效日 / O3_年均_ppb / NO2_年均_ppb / Ox_年均_ppb` | annual mean O3 / its valid days / O3, NO2 and Ox (NO2 + O3) annual means in ppb |

## Notes

One row per station and year. The completeness threshold (274 or 324 valid days) is applied by the analysis scripts, separately for each pollutant, not when the table is built.
