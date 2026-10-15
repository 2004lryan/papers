# Data card: City meteorology, 2015-2024 (workbook)

| Field | Value |
|---|---|
| File (place it at) | `outputs/29_气象构建过程.xlsx`, sheet `02_城市年值` |
| Built by | code/29_fetch_provincial_meteorology.py |
| Source | NASA Prediction of Worldwide Energy Resources (POWER) daily point service, MERRA-2 based (https://power.larc.nasa.gov); city coordinates from the Open-Meteo Geocoding API (https://geocoding-api.open-meteo.com, location database from GeoNames), supplemented by a manual coordinate table in the script |
| Licence / availability | Public release; cite NASA POWER and MERRA-2. Open-Meteo API data are offered under CC BY 4.0 and the GeoNames location database under CC BY. Not redistributed in this repository; available from the corresponding author on reasonable request. |
| Rows x columns | 3370 x 9 (sheet) |
| SHA-256 | `cadf7cd91a6f2d841a565f6d130db46c23d6b549b5c4c37fdc60c8279e921688` |

## Columns

Column names are kept in Chinese because the scripts read them by name.

| Column(s) | Meaning |
|---|---|
| `城市 / 省级行政区 / 年份` | city / province / year |
| `年平均气温_摄氏度 / 年降水量_毫米 / 年平均风速_米每秒 / 年平均相对湿度_百分比 / 年平均气压_百帕` | annual mean temperature (°C) / annual precipitation (mm) / mean wind speed (m s-1) / mean relative humidity (%) / mean surface pressure (hPa) |
| `有效天数` | valid days |

## Notes

Retrieved at each city's geocoded coordinates and aggregated to annual means (precipitation: annual totals). The analysis reads sheet 02_城市年值 from the workbook written by the retrieval script; the checksum is that of the workbook.
