# Data card: City panel with vehicle stock, 2017-2023

| Field | Value |
|---|---|
| File (place it at) | `data/城市面板-2017_2023.csv` |
| Built by | code/37_build_city_panel.py (2023 values from code/40_transcribe_2023_city_nev.py) |
| Source | Same compiled data package as the provincial stock (Ministry of Public Security registration basis); city air quality and meteorology columns come from earlier steps of the pipeline |
| Licence / availability | Third-party compilation (`data/DATA.md` gives its source). Not redistributed in this repository; available from the corresponding author on reasonable request, subject to the terms of the compilation. |
| Rows x columns | 2333 x 30 |
| SHA-256 | `10876f2b75a9d40492fe04ad1ea44d70e04408e2dfd013e7767771c08fb937d7` |

## Columns

Column names are kept in Chinese because the scripts read them by name.

| Column(s) | Meaning |
|---|---|
| `城市 / 年份 / 省级行政区` | city / year / province |
| `NEV保有量_辆 / ln_NEV` | city NEV stock in vehicles / its natural logarithm |
| `本省当年未分配占比` | share of the provincial stock not allocated to any city |
| `PM25年均, PM10年均, O3_8h_90分位, NO2年均, SO2年均, CO年均 and their 有效天 / _有效年 columns` | city annual air-quality values with their valid-day counts and validity flags |
| `年平均气温_摄氏度 / 年降水量_毫米 / 年平均风速_米每秒 / 年平均相对湿度_百分比 / 年平均气压_百帕 / 气象有效天数` | annual mean temperature (°C) / annual precipitation (mm) / mean wind speed (m s-1) / mean relative humidity (%) / mean surface pressure (hPa) / valid meteorology days |

## Notes

The 2023 city values were transcribed from table images in the package and checked against its provincial totals; one missing city was inferred from its province's total.
