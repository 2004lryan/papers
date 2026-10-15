# papers

Code, data cards and reproducibility materials for the papers of
[Ryan Lin](https://2004lryan.github.io) and co-authors.
Each paper has its own top-level folder, named with its three-digit project
number followed by a short name.

| Folder | Paper | Status |
|---|---|---|
| [009within-city-no2-gradients](009within-city-no2-gradients/) | Within-city nitrogen dioxide exposure gradients in China narrowed ahead of electric vehicle fleet growth | In preparation |
| [015calibration-transfer-label-economics](015calibration-transfer-label-economics/) | Calibration transfer under limited target labels: a five-benchmark comparison of correction and representation methods | Submitted to *Chemometrics and Intelligent Laboratory Systems* |
| [016apple-nir-crossyear](016apple-nir-crossyear/) | Evaluation leakage through learnt components in cross-domain Vis-NIR classification: apple origin and five public data sets | — |
| [020nir-reference-error-budget](020nir-reference-error-budget/) | Within-fruit variability of reference soluble solids sets a ceiling on near-infrared calibration performance in apple | In preparation |
| [023arrhenius-recoverability](023arrhenius-recoverability/) | A recoverability trichotomy for Arrhenius inversion: critical design scaling and its consequences for experimental design | In preparation |

## Folder contents

Each folder is self-contained. Its `README.md` explains what the code
reproduces, which data it needs and how to run it. The `LICENSE` and citation
details inside a folder apply to that folder.

## Getting the code

Whole repository:

```bash
git clone https://github.com/2004lryan/papers.git
cd papers/<folder>
```

A single folder only:

```bash
git clone --filter=blob:none --sparse https://github.com/2004lryan/papers.git
cd papers
git sparse-checkout set <folder>
cd <folder>
```

## Citing

Please cite the paper. The reference is given in the folder's README or
`CITATION.cff`.
