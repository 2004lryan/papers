# Functional regression modelling of soluble solids content in apple and hyperparameter selection with small calibration sets

Code and reproducibility materials for the manuscript
**"Functional regression modelling of soluble solids content in apple and hyperparameter selection with small calibration sets"**
(苹果可溶性固形物含量的函数型回归建模与小样本超参数选择).

## Overview

Spectral calibration of soluble solids content (SSC) in apple often has a small
training set and has to be recalibrated for a new batch. This repository holds the
analysis of near-infrared transmittance spectra (590.7–1 249.6 nm, 2 048 points) of
Fuji apples from three producing areas and two years; 1 111 samples remain after
the cleaning step of `code/59data_clean.py`.

Two functional models, functional penalized linear regression (FPLR) and functional
principal component regression (FPCR), both on a cubic B-spline basis, are compared
with PLSR, SVR, random forest and a 1D-CNN under one protocol: the same
preprocessing, the same data splits, nested cross-validation with 5 folds repeated
5 times, and 12 candidate configurations per method. The repository also contains
the small-sample experiment (training sets of 20–400 samples), the cross-batch
recalibration experiment (five batch settings, 5–40 target samples) and the
regression coefficient function of the deployed FPLR model.

Main results, as reported in the manuscript:

- **Unified protocol (Table 2, Fig. 3).** SVR gave an R² of 0.853 and differed from
  PLSR after Holm adjustment. FPLR gave 0.836, with no difference detected from
  PLSR or SVR. Scripts: `code/45rebuttal_R1_4_repeated_ncv.py`,
  `code/47rebuttal_R1_1_paired_tests.py`.
- **Small training sets (Fig. 4).** With 50–200 training samples, FPLR with
  cross-validated selection gave the highest median R² (0.623–0.781). With 30 or
  fewer, every method gave R² < 0 in some draws. Scripts:
  `code/21smallsample_repeats.py`, `code/49rebuttal_derived_stats.py`.
- **Cross-batch recalibration (Fig. 5).** The effect depended on the batch setting.
  For 2018→2019, intercept correction with 10 target samples gave a median R² of
  0.521–0.621; with Xinjiang as the target area, linear corrections left it at or
  below 0.43. Script: `code/46rebuttal_R1_3_recalibration.py`.
- **Batches and coefficient function (Table 1, Figs. 2 and 6).** The six batches had
  mean SSC of 11.84–15.49 °Brix. After preprocessing, the mean spectra of Shandong and
  Shaanxi nearly coincided and those of Xinjiang differed from both. The coefficients
  of FPLR were largest at 868–955 nm and 1079–1115 nm, and only the first window had
  an appreciable influence on the prediction (post hoc description). Scripts:
  `code/70batch_overview.py`, `code/71beta_contribution.py`.

## Getting the code

This folder is `022apple-ssc-functional-regression/` in the repository
[2004lryan/papers](https://github.com/2004lryan/papers/tree/main/022apple-ssc-functional-regression):

```bash
git clone https://github.com/2004lryan/papers.git
cd papers/022apple-ssc-functional-regression
```

In this README, "this repository" and "the repository root" mean this folder; run every
command below from it.

## Repository structure

```
├── code/
│   ├── 59data_clean.py, data_clean.py        # four cleaning rules, row-by-row removal list (Section 1.1, Appendix A9)
│   ├── 25nested_cv.py                        # single-run nested cross-validation; reference that script 47 compares fold by fold
│   ├── 45rebuttal_R1_4_repeated_ncv.py       # nested cross-validation, 5 folds × 5 repeats, per method
│   ├── 47rebuttal_R1_1_paired_tests.py       # Table 2: corrected repeated k-fold t test, Holm adjustment, intervals
│   ├── 54rebuttal_R1_4_boundary_shift.py     # Appendix A6: comparisons on the shifted grid
│   ├── 10smallsample_hyperparam.py, 19svr_cv_aligned.py   # small-sample means that script 21 reconciles its draws with
│   ├── 21smallsample_repeats.py              # small-sample experiment, 20 draws per training set size
│   ├── 18svr_align_and_sensitivity.py, 20cv_selection_mechanism.py   # Appendix A2: SVR input scaling, selection by cross-validation
│   ├── 49rebuttal_derived_stats.py           # derived statistics of the small-sample, recalibration and coefficient-function results
│   ├── 46rebuttal_R1_3_recalibration.py      # cross-batch recalibration, 200 draws per setting and k
│   ├── 48rebuttal_R1_13_bspline_beta.py      # regression coefficient function of the deployed FPLR model (Fig. 6)
│   ├── 56rebuttal_smallsample_selection.py   # Appendix A8: selection-stage rules
│   ├── 57rebuttal_smallsample_guard.py       # Appendix A8: prediction-time safeguards
│   ├── 58rebuttal_guard_posthoc.py           # Appendix A8: post hoc description of the leverage fallback
│   ├── 61rebuttal_mainfold_guard.py          # Section 2.2 and Appendix A8: anatomy of the weak fold
│   ├── 55rebuttal_A5_gbids.py                # Appendix A5: composite index LLH-GBID
│   ├── 51rebuttal_lams_sensitivity.py        # Appendix A3: smoothing parameter of the basis expansion
│   ├── 26basis_nested.py, 27prep_nested.py, 36prep_leakage.py, 38nested_prep.py, 44fully_nested.py
│   │                                         # Appendix A3: basis settings, preprocessing, three-level fully nested recheck
│   ├── 70batch_overview.py                   # Table 1 and Fig. 2: SSC statistics and mean spectra of the six batches
│   ├── 71beta_contribution.py                # Section 2.5 and Appendix A7: decomposition of the prediction by waveband
│   ├── 50rebuttal_figures.py                 # Figs. 3–5 (FIG_LANG=en for the English versions)
│   ├── 40figure_flow.py                      # Fig. 1: workflow and evaluation protocol
│   ├── 42recompute_manuscript_numbers.py, ms_numbers.py   # recompute the numbers quoted in the manuscript from the result files
│   ├── grids.py, grids_boundary_shift.py     # candidate configurations of every method
│   ├── ncv_manifest.py, results47.py, results54.py, export_utils.py, figstyle.py   # shared modules
│   ├── 04baseline_models.py, 08fda_smallsample.py, 16unified_comparison.py   # shared blocks loaded by the scripts above
│   └── MANIFEST.sha256                       # checksum of every file in code/
├── data/DATA.md                # data card: dataset provenance, availability, SHA-256
├── requirements.txt            # pinned to the environment that produced the numbers
├── CITATION.cff
└── LICENSE
```

The scripts in `code/` that are not listed above (all numbered 41 or below) belong to earlier stages of
the project. `42recompute_manuscript_numbers.py` reads none of their outputs.

The scripts address one another and their output folders by the paths of the
authors' working project (`02code/`, `04outputs/`, `05logs/`), and several of them
load a shared block from `08fda_smallsample.py` or `16unified_comparison.py` with
`exec` (`27prep_nested.py` loads its block from `04baseline_models.py`). Run them from the repository root after the one-time preparation given
under Reproduction, not from inside `code/`.

Comments and docstrings in the scripts are in Chinese, and they refer to tables,
figures and sections by the numbering the manuscript had when each script was
written. `40figure_flow.py`, `42recompute_manuscript_numbers.py`,
`70batch_overview.py` and `71beta_contribution.py` use the current numbering.
Scripts 45–48, 50 and 59 use that of the previous version of the manuscript, which
maps to the current one as follows.

| In those scripts | In the current manuscript |
|---|---|
| Table 1 (method comparison) | Table 2 |
| Figs. 1, 2, 3 | Figs. 3, 4, 5 |
| Figs. A1, A2 | Figs. 1, 6 |
| Sections 1.1–1.3 (models) | Sections 1.3.1–1.3.3 |
| Sections 2.1, 2.2, 2.3, 2.4, 2.5 | Sections 1.1, 1.2, 1.4, 1.5, 1.6 |
| Sections 3.1–3.4 | Sections 2.2–2.5 |
| Section 4 (conclusions) | Section 3 |

Tables A1–A7 and Appendices A1–A9 keep their numbers. Scripts of earlier stages,
such as `21smallsample_repeats.py`, use still older numbering.

## Installation

```bash
pip install -r requirements.txt   # Python 3.13
```

PLSR, SVR, FPLR and FPCR were run on CPU (macOS, arm64). The random forest and the
1D-CNN runs of the nested cross-validation were run on a remote server (the 1D-CNN
on GPU); the header of `requirements.txt` lists that environment.

## Reproduction

The dataset is restricted and is not shipped here (see below). Once you have it,
prepare the layout the scripts expect and run them from the repository root:

```bash
export APPLE_SSC_DATA=/path/to/the/two/workbooks
ln -s code 02code                  # the scripts refer to their own folder as 02code/
mkdir -p 04outputs/figs 05logs     # result folder and log folder
export OMP_NUM_THREADS=2           # thread setting of the local runs in the study
export PANEL_AUDIT=off             # export the figures without the panel-alignment check (see the note on the figure scripts)

python 02code/59data_clean.py                    # cleaning list and summary (see the note below)

# Table 1, Fig. 1 and Fig. 2
FIG_LANG=en python 02code/70batch_overview.py
FIG_LANG=en python 02code/40figure_flow.py

# Table 2 and Fig. 3
NCV_METHODS=PLSR,SVR,FPCR,PFLR NCV_SEEDS=20041210 python 02code/25nested_cv.py   # reference results read by script 47
RNCV_BASIS=bspline RNCV_METHODS=PFLR,FPCR python 02code/45rebuttal_R1_4_repeated_ncv.py
RNCV_BASIS=fourier RNCV_METHODS=PFLR,FPCR python 02code/45rebuttal_R1_4_repeated_ncv.py   # sensitivity to the basis
RNCV_METHODS=PLSR python 02code/45rebuttal_R1_4_repeated_ncv.py
RNCV_METHODS=SVR  python 02code/45rebuttal_R1_4_repeated_ncv.py
RNCV_METHODS=RF RNCV_JOBS=2 python 02code/45rebuttal_R1_4_repeated_ncv.py                # server CPU in the study
RNCV_METHODS=1D-CNN RNCV_DEVICE=cuda python 02code/45rebuttal_R1_4_repeated_ncv.py       # server GPU in the study
python 02code/47rebuttal_R1_1_paired_tests.py    # see the note on its provenance check

# Fig. 4, Fig. 5 and the coefficient function (Fig. 6)
python 02code/10smallsample_hyperparam.py        # the two tables script 21 reconciles with
python 02code/19svr_cv_aligned.py
python 02code/21smallsample_repeats.py
python 02code/46rebuttal_R1_3_recalibration.py
FIG_LANG=en python 02code/48rebuttal_R1_13_bspline_beta.py
python 02code/49rebuttal_derived_stats.py
FIG_LANG=en python 02code/50rebuttal_figures.py

# decomposition of the prediction by waveband (Section 2.5, Appendix A7)
python 02code/71beta_contribution.py
```

Where symbolic links are not available, copy `code/` to `02code/` instead. The
scripts print their log to standard output; in the working project it was
redirected to a file in `05logs/`. Results are written to `04outputs/`. Script 48
also writes the deployed models and their model cards to `07checkpoints/`, and
`export_utils.py`, which it imports, creates the folders `03data/processed/`,
`07checkpoints/`, `refine-logs/` and `research-wiki/` at the repository root.
Random seeds are fixed inside each script. In the code, the method the manuscript
calls FPLR keeps its earlier identifier `PFLR`.

Order matters. `70batch_overview.py` and `40figure_flow.py` read the files written
by `59data_clean.py`. `47rebuttal_R1_1_paired_tests.py` reads the results of script
45 for all six methods and `04outputs/25nested_cv.csv`, which `25nested_cv.py`
writes (with the two variables above it runs the four local methods on the seed
that script 47 compares; without them it runs all six methods on five seeds). `21smallsample_repeats.py` writes its table and then reads the tables of
`10smallsample_hyperparam.py` and `19svr_cv_aligned.py` to reconcile the means.
`48rebuttal_R1_13_bspline_beta.py` reads the FPLR results of script 45,
`49rebuttal_derived_stats.py` reads the outputs of scripts 21, 46 and 48,
`50rebuttal_figures.py` reads the outputs of scripts 21, 46, 47 and 49, and
`71beta_contribution.py` reads the output of script 48.

In the study the random forest was run on the CPU of a remote server and the 1D-CNN
on its GPU (script 45 exits if the 1D-CNN is selected without `RNCV_DEVICE=cuda`, unless
`RNCV_SANITY_CPU_CNN=1` is set for a CPU smoke check whose output script 47 does not read).
Both were launched, split into shards and collected by helper shell scripts of the
working project (`02code/deploy45/`), which are not shipped here. The file header
of script 45 describes the sharding variables.

Before it tests anything, script 47 checks the provenance of the output of script
45 from the manifest written with each result file. For PLSR, SVR, FPLR and FPCR
the recorded environment must equal the environment in which script 47 itself is
run. For seed 20041210, the results of PLSR, SVR and the Fourier-basis FPLR and
FPCR must agree fold by fold with `25nested_cv.csv` (the reference run used the
Fourier basis). For the random forest and the 1D-CNN the recorded environment and
cache fingerprint must equal those of the server runs of the study, which are
constants in script 47 (the server environment is listed in the header of
`requirements.txt`). The cache fingerprint covers the data, the candidate grids,
the library versions and the text of the shared block and of `grids.py`. The copy
of `grids.py` shipped here differs from the file used in the study in one line of
its header text (a reference to a document of the working project). A rerun of
these two methods from this repository therefore does not reproduce the
fingerprints of the study, and script 47 does not accept it. If a method is missing
or does not pass, script 47 exits without writing test results. With
`PT_ALLOW_PARTIAL=1` it writes the table and the tests for the methods that pass
to `04outputs/47partial/`; from this repository these are PLSR, SVR, FPLR and
FPCR. `50rebuttal_figures.py` takes the complete output of script 47 when it is
valid and the partial output otherwise, and exits when there is neither. It also
exits, before it draws any figure, when the output of script 47 records a method
that did not pass the check. A random forest or 1D-CNN run completed from this
repository is such a method, for the reason given above. To draw Figs. 3–5 from
this repository, run script 47 with `PT_ALLOW_PARTIAL=1` while `04outputs/` holds no
result files of these two methods (`45rebuttal_ncv_none_RF*` and
`45rebuttal_ncv_none_1DCNN*`): skip the two server commands, or move their output
out of `04outputs/` first. The figures are then drawn from the partial output.

`59data_clean.py` writes the row-by-row cleaning list and the summary. Its last
step collects the results obtained before cleaning, which Appendix A9 quotes, from
an earlier commit of the authors' working repository with `git show`. That commit
is not part of this repository, so here the script stops with a `git` error at
that step, after the two files have been written. The cleaning rules themselves
are in `data_clean.py`. They are applied wherever the data are loaded through the
shared blocks `08fda_smallsample.py` and `16unified_comparison.py` and in scripts
04, 46, 59 and 70. The pilot scripts 01, 02, 03, 07, 09 and 11 read the workbooks
directly and only drop records with SSC outside 5–25 °Brix. Scripts 05 and 06
read the 2025 batch, which the manuscript does not use (see `data/DATA.md`).

The figure scripts (`40figure_flow.py`, `48rebuttal_R1_13_bspline_beta.py`,
`50rebuttal_figures.py`, `70batch_overview.py`) check the alignment of multi-panel figures with a module,
`audit_panel_alignment.py`, that is not shipped here. Set `PANEL_AUDIT_DIR` to the
directory that holds it, or set `PANEL_AUDIT=off` to export the figures without the
alignment check.

`code/42recompute_manuscript_numbers.py` recomputes the quantities quoted in the
manuscript from the result files and compares them with the text of the
manuscript file, which is not shipped here.

Comments in the scripts also cite documents of the working project that are not
shipped: the project rules (cited as 项目规范 or by a bare chapter number such as
7.1), the preregistered protocol (协议 or 预注册, items B10.x), the submission
checklist (投稿清单) and the comments of the corresponding author on an earlier
draft (导师意见, 批注), the project proposal (申报书) and a study of papers of the target
journal (`research-wiki/目标期刊论文/00VENUE_STUDY.md`). They also name helper scripts and logs of the working
project that are not shipped, such as the shell scripts under `02code/deploy45/`,
`35export_code.py` and files under `05logs/`. A path written as
`08github/data/DATA.md` in a comment is `data/DATA.md` of this repository.

The header of `grids.py` describes how the main grid was derived from the initial
grid. Appendix A1 of Online Resource 1 is the authoritative account. On the initial
grid the random forest selected 200 trees in one of the five folds, the 1D-CNN
selected a dropout of 0.4 in two folds and FPCR selected `L = 40` in one fold;
these three values are not in the main grid, so rule ④ of that header and the
comment on the random-forest line do not hold for them. The initial-grid run used
the Fourier basis for the two functional models. The header is left as written,
because the text of the `grids.py` used in the study enters the cache fingerprint
of the results.

### Fonts

The figure scripts request Arial for Latin text and Source Han Sans CN for Chinese
text. When these fonts are not installed, matplotlib falls back to its default
font. `70batch_overview.py`, `48rebuttal_R1_13_bspline_beta.py` and
`50rebuttal_figures.py` then still export their figures, with the fallback font.
On a machine without the two fonts, `40figure_flow.py` exited with code 1 at its
check for overlapping or out-of-bounds text. Fig. 1 of the manuscript was exported
with Arial; install the two fonts before running it.

### Check on a second platform

PLSR, SVR, FPLR and FPCR were computed on macOS (arm64) in the study. The commands
above were also run once from a clean copy of this package on Linux (x86-64,
Python 3.13.5, the library versions of `requirements.txt`), with `MPLBACKEND=Agg`
and `KMP_DUPLICATE_LIB_OK=TRUE` set in addition to the variables above. The two
server commands were not run to completion: the random forest command was stopped
after 600 s, before any seed had finished, and the 1D-CNN command was run only
without `RNCV_DEVICE=cuda`, where script 45 exits. Script 47 therefore exited
without output when run as listed and wrote its partial output with
`PT_ALLOW_PARTIAL=1`. `59data_clean.py` stopped at its `git show` step after
writing its two files, and `40figure_flow.py` exited as described under Fonts. The
other commands ran to completion.

After the run, 23 result tables were compared cell by cell with the files of the
study: the two files of script 59, the three tables of script 70,
`25nested_cv.csv`, the six fold-level tables of script 45 for the four CPU methods
(without their `fingerprint` column, which differs by construction, see the note
on the provenance check), the tables of scripts 10, 19 and 21, the summary and
the draws of script 46, two of the three tables of script 48 (`48rebuttal_beta.csv` and
`48rebuttal_beta_windows.csv`), three tables of script 49 and
`71beta_contribution.csv`. The other files that the scripts write were not
compared, among them the per-sample predictions of scripts 25 and 45,
`46rebuttal_recal_batches.csv`, `48rebuttal_deploy.csv`, `49rebuttal_smallsample_paired.csv`, the two other
tables of script 71 and the figures. 5 of the 23 files were byte-identical
(`59data_clean_list.csv`, `59data_clean_summary.csv`, `70batch_stats.csv`, `48rebuttal_beta_windows.csv`, `49rebuttal_recal_range.csv`), and 5 more had no cell with a relative
difference above 1e-9 (`70batch_spectra_summary.csv`, `45rebuttal_ncv_bspline_FPCR.csv`, `45rebuttal_ncv_none_PLSR.csv`, `10smallsample_hyperparam.csv`, `49rebuttal_beta_band.csv`).
The other 13 files had such cells:

| File | Numeric cells | Cells with relative difference above 1e-9 | Largest absolute difference (column) |
| --- | --- | --- | --- |
| `70batch_mean_spectra.csv` | 26624 | 10 | 9.02e-16 (`prep_2018_新疆`) |
| `25nested_cv.csv` | 245 | 36 | 0.000217 (`P90`) |
| `45rebuttal_ncv_bspline_FPLR.csv` | 375 | 1 | 1.99e-10 (`P90`) |
| `45rebuttal_ncv_fourier_FPLR.csv` | 375 | 21 | 5.61e-09 (`P90`) |
| `45rebuttal_ncv_fourier_FPCR.csv` | 375 | 1 | 1.3e-12 (`P90`) |
| `45rebuttal_ncv_none_SVR.csv` | 375 | 175 | 0.000288 (`P90`) |
| `19svr_cv_aligned.csv` | 21 | 14 | 9.95e-05 (`SVR_cv`) |
| `21smallsample_repeats.csv` | 2940 | 261 | 0.00241 (`R2`) |
| `46rebuttal_recal_summary.csv` | 5840 | 1217 | 0.00235 (`R2_lo`) |
| `46rebuttal_recal_draws.csv` | 320000 | 47547 | 0.0877 (`R2`) |
| `48rebuttal_beta.csv` | 10240 | 1844 | 8.52e-05 (`beta_q50`) |
| `49rebuttal_smallsample_summary.csv` | 343 | 69 | 0.00241 (`min`) |
| `71beta_contribution.csv` | 96 | 6 | 5.74e-05 (`median_abs_beta`) |

In `46rebuttal_recal_draws.csv` 64000 non-numeric cells also differed, one in each
row. They are the `fp` column: this fingerprint hashes the interpreter and library
version strings and the text of `grids.py` together with the data, the draws and
the code, so it differs between the two runs by construction. The partial output of
script 47 was compared with the corresponding rows of the complete output of the
study: in its table (6 rows) the largest absolute difference was 7.35e-06
(`RPD_sd`), and in the six pairs of its main analysis 0.000363 (`t`). For these
pairs the nine columns from `n_pairs` to `a_wins` were compared; `shapiro_p`,
`wilcoxon_p`, `holm_p` and `family_size` were not, the last two because they depend
on the number of pairs in the family.
`21smallsample_repeats.py` reconciles the 49 means of its draws, at three decimals,
with the tables that scripts 10 and 19 had written in the same run; none disagreed.
A rerun on another platform can thus differ from the reported numbers in the third
decimal, and single draws of the recalibration experiment can differ by more.

## Code quality

In the working project the scripts pass `ruff check` and `mypy` under the
project's own configuration (`pyproject.toml`), which is not shipped here. The
lint selection there is deliberately narrow: the scripts that load a shared block
through `exec` use names that are defined only at run time.

## Data availability

- **Apple NIR transmittance spectra and SSC (2018 and 2019 batches)** were measured
  by the research group at Xinjiang Agricultural University and are **not publicly
  released**. They are available from the corresponding author on reasonable
  request. This repository ships the data card and SHA-256 checksums
  (`data/DATA.md`) but **not** the spectra.
- **No external public datasets are used.** Every number in the manuscript comes
  from the restricted dataset above.

## Citation

See `CITATION.cff`.

## License

Code: **MIT** (see `LICENSE`). The manuscript is **not** included in this
repository.

## Contact

Correspondence and data requests: HUANG Hua — huanghua@xjau.edu.cn
