# Datasets — sources, licenses, and availability

This repository ships **data cards and checksums only**, never the spectra. The
dataset behind every number in the manuscript is self-collected and restricted;
obtain it from the corresponding author (see below), place the two workbooks in
one directory, and point the analysis scripts at it with

```bash
export APPLE_SSC_DATA=/path/to/that/directory
```

All scripts read that variable and fall back to `../../05data/001_apple_hyperspectral_multiyear`.

## Restricted / self-collected data (NOT in this repo)

| Dataset | Modality | Availability | SHA-256 |
|---|---|---|---|
| Apple NIR transmittance + SSC, 2018 batch | 2 048-band transmittance spectra, 590.7–1 249.6 nm; destructive SSC reference (°Brix) | **Restricted** — collected by the research group at Xinjiang Agricultural University; not publicly released; available from the corresponding author (HUANG Hua, huanghua@xjau.edu.cn) on reasonable request | `051f0725274fceea2dc402f5b401ccbdf03fe6f98a76d49836ac09fb90332465` |
| Apple NIR transmittance + SSC, 2019 batch | same instrument and protocol as 2018 | **Restricted** — same terms | `fc1b38299b2e526a5cb851b5e53d81257d1d5f9f8a7595d7b3551343b9061afe` |

File names (kept verbatim, Chinese): `01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx`
(10 839 203 bytes) and `1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx` (12 287 828 bytes).

### What the analysis actually uses

| Item | Value |
|---|---|
| Cultivar | Fuji apple (*Malus domestica* cv. Fuji), single cultivar |
| Origins | Aksu (Xinjiang), Yantai (Shandong), Luochuan (Shaanxi) |
| Sample source | collected by the research group in 2018 and 2019 (manuscript, Section 1.1) |
| Batches | 2018 and 2019 (the year in the file name of each workbook) |
| Raw records | 1 301 (2018: 630; 2019: 671) |
| Records removed before modelling | 190, the union of four rules applied separately within each year (`code/data_clean.py`): spectral outlier rule (Euclidean distance to the yearly mean spectrum, after standardizing each wavelength within the year, above the 90% quantile; 130 records), SSC anomaly rule (SSC above 20 °Brix; 1 record), duplicate spectrum rule (raw spectra identical at all 2 048 points but with different SSC, removed as whole groups; 34 groups, 69 records), detector saturation rule (a reading of 65 535 at any wavelength; 2 records, both also flagged by the spectral outlier rule) |
| Samples used | **1 111** (2018: 537; 2019: 574; Xinjiang 364, Shandong 357, Shaanxi 390) |
| Row-by-row removal list | written by `code/59data_clean.py` (rule hits, distance under the spectral outlier rule, sample numbers before and after cleaning) |
| Results before cleaning | 1 300 samples (only the record with SSC outside 5–25 °Brix removed); reported for comparison in Appendix A9 of Online Resource 1 |
| Spectral range | 590.7–1 249.6 nm, 2 048 points |
| SSC reference | ATAGO PR-101 digital refractometer (Japan), destructive measurement |
| SSC distribution | 13.40 ± 1.68 °Brix (2018: 12.89 ± 1.51; 2019: 13.88 ± 1.70) |
| Preprocessing in code | SNV followed by Savitzky–Golay first derivative (window 15, order 2) |

A third acquisition year (2025) exists in the same source directory but is **not
used** anywhere in this manuscript: it was collected with a different origin set
(Gansu replaces Shaanxi), a different instrument and spectral range (reflectance
at 346 bands, 391.1–1 001.1 nm) and a different file layout, so it is not
comparable under the single protocol this paper insists on. In an exploratory
check at the start of the project (`code/05check_2025_scale.py`,
`code/06diagnose_2025.py`), none of the settings tried predicted SSC on this
batch (R² from −0.199 to 0.115); the cause has not been identified. No result
in the paper depends on it.

### Ethics and licensing

The material is fruit; no human or animal subjects are involved and no ethical
approval applies. The data contain no personal information. Redistribution of
the raw workbooks is not permitted; the checksums above let a recipient verify
that the copy they obtained is the one the reported numbers were computed from.

## External public datasets

**None.** Every number in the manuscript comes from the restricted dataset above.
