# Data card: Provincial new energy vehicle stock, 2017-2023

| Field | Value |
|---|---|
| File (place it at) | `data/省级新能源汽车保有量-公安部口径-2017_2023.csv` |
| Built by | code/32_integrate_nev_stock.py |
| Source | Year-end registered new energy vehicles on the Ministry of Public Security registration basis, from a third-party compiled data package; national totals are published in the ministry's annual bulletins (https://www.mps.gov.cn) |
| Licence / availability | Third-party compilation (`data/DATA.md` gives its source). Not redistributed in this repository; available from the corresponding author on reasonable request, subject to the terms of the compilation. |
| Rows x columns | 217 x 5 |
| SHA-256 | `36f46632fc39ec649be3536cc1b4194f28cff87815912be6a6a818c64d4fa4d9` |

## Columns

Column names are kept in Chinese because the scripts read them by name.

| Column(s) | Meaning |
|---|---|
| `省级行政区 / 年份` | province / year |
| `新能源汽车保有量_辆 / 新能源汽车保有量_万辆` | NEV stock in vehicles / in 10,000 vehicles |
| `新能源汽车年增量_万辆` | annual increase, 10,000 vehicles |

## Notes

The provincial sum lies within 1% of the published national total in every year from 2017 to 2023.
