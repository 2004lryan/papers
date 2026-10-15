# Data card: Built-up exposure at 500 m and 1000 m

| Field | Value |
|---|---|
| File (place it at) | `data/站点建成区暴露.csv` |
| Built by | code/45_station_builtup_exposure.py |
| Source | Global Human Settlement Layer, built-up surface, release R2023A (European Commission, Joint Research Centre), https://human-settlement.emergency.copernicus.eu |
| Licence / availability | CC BY 4.0; cite the reference publication of the GHSL release. Not redistributed in this repository; available from the corresponding author on reasonable request. |
| Rows x columns | 1601 x 11 |
| SHA-256 | `3acbc9f46a30fe6cc61e30516c1fd02387d07028b0109695a7afc53c27b52c00` |

## Columns

Column names are kept in Chinese because the scripts read them by name.

| Column(s) | Meaning |
|---|---|
| `站点编号 / 站点名称 / 所属城市 / 纬度 / 经度` | station code / name / city / latitude / longitude |
| `瓦片` | GHSL tile identifier |
| `建成区面积_m2_500m / 建成区占比_500m` | built-up area (m2) / built-up fraction within 500 m |
| `建成区面积_m2_1000m / 建成区占比_1000m` | built-up area (m2) / built-up fraction within 1000 m |
| `建成区占比_城内百分位` | within-city percentile rank of the 500 m built-up fraction |

## Notes

Computed from the 100 m built-up surface raster around each registered station. Limitation: the built-up fraction measures position in the urban core, which also tracks population and commercial activity, not traffic volume alone. The manuscript tests its traffic signature for NO2 with the commuting-peak to overnight ratio, the weekday excess and O3 titration, and notes that the axis is not specific to traffic because SO2 also rises along it.
