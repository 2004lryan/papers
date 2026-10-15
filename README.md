# Recoverability trichotomy for Arrhenius inversion

Code and reproducibility materials for the manuscript

> **A recoverability trichotomy for Arrhenius inversion: critical design scaling and its consequences for experimental design**
> (in preparation for submission to *Chemometrics and Intelligent Laboratory Systems*)

## What the paper claims

When activation energies are inverted from a narrow temperature window, the Arrhenius
parameters become almost indistinguishable along one direction and extra sampling does not
repair it. Recoverability is governed by one computable scalar

$$\Psi_n = n\,\lambda_{\min}(\hat I_1),$$

the sample size times the smallest eigenvalue of the normalised Fisher information, and it is
strictly trichotomous: as the design window shrinks like a power $\rho$ of the sample size, the
error vanishes at rate $\Psi^{-1/2}$ for $\rho<1/2$, is constant at $\rho=1/2$, and stays bounded
away from zero for $\rho>1/2$. The degenerate direction is the classical kinetic compensation line.

**What we do not claim**, and the code shows why:

- $\Psi$ is, as a criterion, classical **E-optimality**; no novelty is claimed for that.
- The branch $\Psi\to\infty$ is the established least-squares consistency criterion.
- The compensation-line correspondence is an identity.
- The design prescription (widen the window) is not new as a prescription.
- Three claims of ours are **falsified in the paper itself**, by these scripts:
  $\Psi$ has no discriminating power among $\Psi$-equivalent designs; a D-optimality
  comparison we withdrew; and $\Psi$ as a *calibration* diagnostic (`124coverage_study.py`).

## Repository layout

```
code/       analysis scripts, numbered as in the manuscript
outputs/    the result files behind the paper's numbers, and a few bookkeeping and audit records
data/       extracted source tables (CC BY 4.0) + SHA-256 manifest
logs/       the RYA-103 independent-rerun archive
figures/    the four main-text figures
docs/       the data card
tools/      bootstrap, manifest generator, pre-push privacy guard
```

The scripts resolve their inputs and outputs against the **numbered** directory names of the
working tree in which the published results were produced (`02code/`, `03data/`, `04outputs/`,
`05logs/`, `06doc/`), and `data/SHA256-MANIFEST.tsv` records those same paths. They are the
working scripts as they stand and are deliberately not rewritten for the shipped layout: a
reproducibility mirror whose code differs from the working code is worth less than one that
needs a documented bootstrap step. `tools/bootstrap_layout.py` reconciles the two by
creating the numbered names as aliases. On Linux and macOS the aliases are committed symlinks,
so a fresh clone already works and the bootstrap only verifies; on Windows, git checks symlinks
out as text stubs unless `core.symlinks=true`, and the bootstrap replaces them with directory
junctions.

Some scripts had comments, docstrings or descriptive text strings edited after the published
numbers were produced. For three of them, `103`, `143` and `145`, the shipped result files were
not regenerated after that edit; the text fields concerned were rewritten in place instead, and
`outputs/155relabel_release_fields.json` lists each one.

## Reproduction

```bash
pip install -r requirements.txt
python tools/bootstrap_layout.py               # verify/repair the layout aliases — run this first

python code/123montecarlo_criterion_study.py   # 1000 independent designs
python code/124coverage_study.py               # CI coverage study
python code/122criterion_bakeoff.py            # E / D / A / c comparison
python code/126trichotomy_plugin_psi.py        # plug-in Psi(theta-hat), no oracle
python code/133ablation_innovation_map.py      # innovation-to-ablation map
python code/120datacard_sha256.py              # regenerate the data card
python tools/make_manifest.py                  # regenerate data/SHA256-MANIFEST.tsv from it
```

Run those last two after any of the others, and in that order. Re-running an analysis script
rewrites its artefact, so until the data card is regenerated `tools/make_manifest.py` will refuse
to write — it will not record a hash it cannot verify against the bytes on disk.

All runs are single-threaded CPU, seeded, and complete in minutes to tens of minutes; no GPU is
used. `123montecarlo_criterion_study.py` is the long one (of order ten minutes; see
`outputs/128montecarlo_runtime_receipt.json` for measured wall-clock and CPU time).

### What runs from this repository alone, and what does not

Stated explicitly, because a reproduction package that quietly needs files it does not ship is
worse than one that says where the boundary is.

Runs here with no further inputs: `43trichotomy_formal`, `54c5_realkinetics`, `59c5_supp_table`,
`60fig_realc5`, `91c5_realkinetics_killer`, `99_realkinetics2`, `103sensitivity_pilot`,
`104ablation_psi_aggregation`, `106negctrl_permutation`, `107design_multivariety`,
`119pazman_pronzato_bridge`, `120datacard_sha256`, `122criterion_bakeoff`,
`123montecarlo_criterion_study`, `124coverage_study`, `126trichotomy_plugin_psi`,
`127budget_matched_sensitivity`, `130aggregate_pilot_robustness`,
`131aggregate_c5_partial_tally`, `133ablation_innovation_map`,
`137design_variance_scale_invariance`, `138eigendirection_angle_check`,
`141rya579_seedscope_receipt`, `142rya579_method_identity_receipt`,
`144rya579_critical_kl_check`.

Needs something this repository does not ship:

| script | needs | why it is not shipped |
|---|---|---|
| `102_avocado_c5_pilot.py`, and `105multiseed_realdata.py` through it | the raw Hass avocado workbook | large third-party dataset, reached through its Mendeley DOI (below); point `ARIS_RAW_ROOT` at the download |
| `121audit_en_assurance.py` | the manuscript `.tex` sources | the manuscript itself, not published here |
| `143rya579_ledger_receipts.py` | the manuscript `.tex` sources and the obligations ledger of the forensic review | it is a ledger bookkeeping receipt over the audit trail, not a result of the paper; neither input is published here |
| `42paper_figs_v8.py` (panels 1, 2, 4) | `04outputs/_from_server_v5/` | a shared figure driver; those inputs belong to a different project line of ours and are outside this paper. The main-text panels it draws are shipped as PDFs in `figures/`. Its Krug panel (`fig3_krug.pdf`) is the exception and **does** run here: since 2026-08-23 that panel reads `outputs/43trichotomy_formal-percell.csv`, which is shipped |
| `55fetch_pmc_tables.py`, `125verify_source_dois.py` | network access | they re-fetch the published source tables and re-check the source DOIs |

Three bookkeeping scripts are not shipped; what they wrote is. `145citation_field_diff.py`
compared the bibliography of an earlier draft field by field with Crossref
(`outputs/145citation_field_diff.json`); its input, the citation-resolution record, was produced
inside the private review sandbox. `132redact_session_logs.py` replaced machine-specific path
prefixes in the RYA-103 archive with placeholders and wrote
`logs/rya103_verify_20260810/REDACTION_LEDGER.md`. `155relabel_release_fields.py` rewrote a few
descriptive text fields, and no numbers, in twelve registered result files;
`outputs/155relabel_release_fields.json` gives for each file the SHA-256 registered before the
rewrite, the SHA-256 after it and the number of fields changed per rule. Both of the last two
take as their baseline the files as they were before, which exist only in the private working
history. None of the three produces a result of the paper.

`128montecarlo_runtime_receipt.py` re-runs `123` three times in a sandbox under `$HOME/.cache`
to time it; it works here but takes roughly half an hour.

Two further things a reader will notice. Script docstrings cross-reference internal project
documents by their working-tree names (design notes, the submission audit, the project brief);
those are editorial records, not inputs, and nothing here depends on them — the one that is a
genuine reference, the data card, is shipped as `docs/0NDATACARD_C5.md`. And several scripts write
an `.xlsx` workbook beside their JSON output; those are excluded from the repository and
`tools/check_no_leak.sh` reports them as reproduction side effects rather than failing.

## Integrity checks

```bash
python tools/make_manifest.py --check     # every manifest row still hashes to what is recorded
python tools/bootstrap_layout.py --check  # every manifest path resolves as written
bash  tools/check_no_leak.sh              # no private paths, raw blobs or credentials
```

`data/SHA256-MANIFEST.tsv` is a projection of `outputs/120datacard_sha256.json`, the data card
emitted by `code/120datacard_sha256.py`; the two are kept one-to-one because a manifest assembled
by hand is a manifest that drifts. It has 99 rows: 98 files present here and verified byte for
byte, plus the one external dataset below, whose SHA-256 is recorded so that a downloaded copy
can be checked. What `make_manifest.py` proves is that every row of the data card is here and
intact; that the card in turn enumerates every dependency of every number is enforced upstream,
by the card's own reverse gate over the result directories and script-family prefixes
(`missing` and `unregistered_in_guarded_dirs` in the JSON, both empty here). `code/` is outside
the manifest, whose scope is the source and derived files the paper's numbers depend on; the
release commit pins it.

One caveat on re-running rather than checking. Within one machine and one BLAS build the run is
bit-reproducible: `outputs/128montecarlo_runtime_receipt.json` records three reruns of
`123montecarlo_criterion_study.py` whose output hashed identically to the shipped artefact.
Across machines or BLAS builds the last one or two significant figures move with reduction
order, so the bytes can differ while the numbers do not. Re-running it here on a different
build gave a largest relative departure of order $10^{-10}$ across the 1000-design table, with
the reported slopes ($-0.486$ overall, medians $-0.505$ and $-0.491$, the $R^2=0.732$ band)
unchanged to every digit the paper prints. So compare numbers after a re-run, and hashes only
against the shipped files.

## Data

No new data were collected. All three real systems are previously published third-party
data under CC BY 4.0 — see [`data/DATA.md`](data/DATA.md) for sources and DOIs, and
[`docs/0NDATACARD_C5.md`](docs/0NDATACARD_C5.md) for the full data card. The two literature
tables are shipped here as extractions; the avocado dataset is large and is referenced
through its Mendeley DOI rather than re-hosted.

**Not in this repository, by design:** the group's own unpublished spectral datasets, which
belong to other projects and are not used by this paper. `tools/check_no_leak.sh` is run before
publishing and exits non-zero if any of them, any raw-data blob, any local absolute path or any
credential appears, in text or inside a binary. One exception is reported rather than failed:
three hash-registered provenance files, the aggregate result and the runner of the RYA-103
archive and `requirements.lock.txt`, record the absolute path of the interpreter used for the
run, which is what they are for, and redacting them would invalidate the published manifest.
The script prints that count on every run.

## Citation

See [`CITATION.cff`](CITATION.cff). A BibTeX entry will be added on acceptance.

## License

Code: MIT (see [`LICENSE`](LICENSE)). Each dataset remains under its own licence; please
cite the original sources.

## Contact

Hua Huang (corresponding author) — huanghua@xjau.edu.cn — Xinjiang Agricultural University.
