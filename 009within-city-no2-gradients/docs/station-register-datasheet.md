# Data card: Station register with coordinates

| Field | Value |
|---|---|
| File (place it at) | `data/国控站点元数据.csv` |
| Built by | code/43_fetch_station_metadata.py |
| Source | CNEMC real-time publication platform (https://air.cnemc.cn:18007/) |
| Licence / availability | Public release by CNEMC. Not redistributed in this repository; available from the corresponding author on reasonable request. |
| Rows x columns | 1601 x 7 |
| SHA-256 | `29964754dfafcac06390223bff667d2edb847b0ea49bc3843ccbab25bef2eec1` |

## Columns

Column names are kept in Chinese because the scripts read them by name.

| Column(s) | Meaning |
|---|---|
| `站点编号 / 站点名称` | station code / station name |
| `所属城市 / 城市代码 / 省份ID` | city / city code / province identifier |
| `纬度 / 经度` | latitude / longitude |

## Notes

The register lists stations that are currently published; stations absent from it have no coordinates and cannot enter the panel.
