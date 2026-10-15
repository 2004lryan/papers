# Manuscript-side directory

This directory stands for the manuscript folder of the working tree and is shipped empty on
purpose.

- `code/42paper_figs_v8.py` and `code/60fig_realc5.py` write the main-text figures to
  `figures_v8/` here and create that folder on first run. The figures used in the paper are
  shipped in [`../../figures/`](../../figures/).
- `code/121audit_en_assurance.py` and `code/143rya579_ledger_receipts.py` read the manuscript
  `.tex` sources from here. The manuscript is not published in this repository, so neither of
  them runs from it; see the top-level README.

Two generated files go to `outputs/` and are not committed, because both are regenerated from the
shipped inputs: `code/120datacard_sha256.py` writes `0NDATACARD_C5_manifest.tsv`, the per-file
SHA-256 listing behind [`../0NDATACARD_C5.md`](../0NDATACARD_C5.md), and
`code/59c5_supp_table.py` writes `supp_c5_tables.tex`.
