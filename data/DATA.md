# Datasets — sources, licenses, and availability

This repository ships **data cards only** (see `../docs/`), not spectra. Obtain
each dataset from its official source below and place it under `data/<id>/`.
SHA-256 checksums (first 16 hex) are those of the files the analysis reads; Supplementary Table S1
(Section S1.6) refers here for them.

## Main benchmark (restricted access — not in this repo)

| Dataset | Modality | Availability | SHA-256 (16) |
|---|---|---|---|
| Apple Vis-NIR (3 acquisition years, 4 declared origins in 9 year-by-origin groups; 1,950 fruits, 4,529 spectra) | 2018 and 2019: fibre-optic transmission spectra; 2025: hyperspectral reflectance imaging; harmonised to 229 bands (591–1001 nm) | **Restricted**: internal data of Xinjiang Agricultural University; **not publicly released**. Available from the corresponding author (huanghua@xjau.edu.cn) on reasonable request. | `f5c0afff9b18c3eb` |

Data cards for the apple benchmark are in `../docs/`.

## External public datasets (download from official source; not re-hosted)

| Dataset | Modality / drift axis | Source | SHA-256 (16) |
|---|---|---|---|
| OliveOil (memorisation ceiling only — split by index, not by domain; excluded from the analysis) | FT-IR, 4 countries | [UCR Time Series Archive](https://www.timeseriesclassification.com/description.php?Dataset=OliveOil) | `472c5b73157b2e0f` |
| Cargill corn | NIR, 3 instruments (m5/mp5/mp6) | [Eigenvector Research](https://eigenvector.com/resources/data-sets/) | `e28fd4be274a54ca` |
| IDRC 2002 tablet ("shootout") | NIR, 2 instruments | [Eigenvector Research](https://eigenvector.com/resources/data-sets/) | `129a32ec9e194e56` |
| Anderson mango | NIR, 4 harvest seasons | [Mendeley Data](https://data.mendeley.com/datasets/46htwnp833) | `86159e2df9244fa5` |
| Wohlers kiwifruit, unit KK1 | NIR, 2017, 2018 and 2019 harvest seasons on one instrument; the analysis keeps the spectra of unit KK1 | [figshare](https://doi.org/10.6084/m9.figshare.21747983) | `37157f2c42a5d441` |
| Indian Pines | AVIRIS hyperspectral, spatial split | [EHU / Purdue](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | cube `ec2f8808710919d5` / gt `65c4687a8ab04f6d` |

Each dataset is used under its own license; please cite the original sources.
`code/fetch_datasets.py --list` prints the same pointers.
