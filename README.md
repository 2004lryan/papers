# Evaluation leakage through learnt components in cross-domain Vis-NIR classification — evaluation code

Evaluation code, data cards and checksums for the manuscript

**"Evaluation leakage through learnt components in cross-domain Vis-NIR classification: apple origin
and five public data sets"**.

## What the paper does

Cross-domain evaluation of a spectral classifier can leak through any learnt component that has seen
the held-out domain: an encoder trained once on all data, or a checkpoint or hyperparameters selected
on held-out spectra. The paper measures this leak on Vis-NIR transmission spectra of 1,301 apples of
three vendor-declared origins, acquired in 2018 and 2019 on one instrument, and on five public
datasets split by instrument (maize, tablets), harvest season (mango, kiwifruit) or spatial quadrant
(Indian Pines): 18 held-out domains in six datasets. Apple origin labels are vendor declarations
collected at purchase, so the apple task is discrimination of *declared* origin. A three-year apple
design that adds hyperspectral reflectance images of 2025, and with them a change of instrument, is
reported in the Supplementary Material.

Three results, each reproduced by scripts in this repository:

1. **An exposed encoder inflates cross-domain results.** Prototypical networks with six encoder
   families (CNN, MLP, ResNet and Transformer, a supervised PLS projection and an unsupervised PCA
   projection) were trained under arms that differ only in which held-out data enter training
   (`64_leakage_controls.py` on the apple data, `74_multi_encoder_leakage.py` on the public
   datasets). Admitting the labelled held-out domain raised macro-F1 at five reference samples per
   class by 0.121, 0.114, 0.096 and 0.051 for the four deep families (mixed-effects means over the 18
   held-out domains, `75_leakage_generality_summary.py`); the change was 0.009 for the PLS projection
   and −0.001 for PCA. On the apple data the exposed CNN reached 0.94 in both folds, against 0.65 and
   0.69 for the source-only encoder; choosing only the checkpoint of the source-only encoder on the
   held-out year added 0.028 (0.005 in the three-year design). Leaving out maize and tablets, whose
   domains share physical samples, or Indian Pines, the only image data set, kept the inflation of the
   deep families, except that without Indian Pines the ResNet interval includes zero.
2. **Correct held-out labels carry the inflation.** Permuting the held-out labels removed the gain,
   and held-out samples other than the evaluated ones produced most of it.
3. **Exposure changes which method is chosen.** Selecting the latent-variable count of partial least
   squares discriminant analysis on the held-out year raised its macro-F1 by 0.055 at five fruits per
   origin (`76_target_tuning_leakage.py`), and by 0.018 on average over the 18 held-out domains.
   Together with the exposed encoder, exposure changed the first-ranked method in 20 of 69
   domain-by-K cells, each time to the prototypical network with the exposed encoder (`72_kscan_arms.py`,
   `74_multi_encoder_leakage.py --route_panel` and `76_target_tuning_leakage.py`, summarised by
   `75_leakage_generality_summary.py`).

## Scripts by manuscript item

| Manuscript item | Script(s) in `code/` |
|---|---|
| Preprocessing: crop to 590–1,001 nm, second-order Savitzky–Golay (15 points) on each instrument's native grid, cubic spline onto 229 bands (591–1,001 nm), SNV (Section 2.4) | `01preprocess.py` |
| Encoder arms on the apple designs, the checkpoint-only arm included (Sections 2.6, 2.7, 3.2 and 3.3, Table 1; seed-level detail with quality-control and smoothing variants, Section S6.3, Table S11; three-year design, Section S7.1, Table S15, Fig. S2) | `64_leakage_controls.py` (with `protonet.py`, `encoders.py`, `44_budget_matched_control.py`) |
| Encoder families and route panel on the public datasets (Sections 2.5, 2.7, 2.8, 3.2 and 3.4) | `74_multi_encoder_leakage.py` (with `encoders.py`, `crossdataset_lib.py`, `fetch_datasets.py`) |
| Mixed-effects means, cluster bootstrap, sign tests, the two sensitivity analyses and changes of the first-ranked method (Sections 2.9 and 3.2–3.4, Table 2, Fig. 3, Tables S16–S18) | `75_leakage_generality_summary.py` |
| Hyperparameters selected on the held-out year (Sections 2.8 and 3.4, Section S8.3, Table S19) | `76_target_tuning_leakage.py` (grids of `71_source_only_tuning.py`) |
| Method comparison across support-set sizes (Section 3.4, Fig. 4; three-year design, Sections S6.1 and S7.2, Table S9, Fig. S3) | `72_kscan_arms.py`, `71_source_only_tuning.py`, `65_domain_adaptation_baselines.py`, `22_cnn_resnet_baseline.py`, `42_ensemble_kshot.py` |
| Source-year hyperparameter selection and grid-edge check (Section S6.6, Table S14) | `71_source_only_tuning.py`, `73_grid_edge_check.py` |
| Quality control and smoothing in spectral space (Section 3.1, Section S6.2, Table S10) | `66_sg_window_sensitivity.py` |
| Nine-route panel at fixed settings (Section S5.6, Table S6) | `20_unified_singleface_kshot.py` |
| Labelled and unlabelled exposure on the public datasets: supervised, masked and contrastive self-supervised (Section S1.6, Table S1) | `24_crossdataset_leakage.py`, `25_mixed_effects_leakage.py`, `31_leakage_decomposition.py`, `35_stronger_ssl_control.py`, `foundation_lib.py` |
| Diagnostics of the sufficient conditions on real embeddings (Section S1.5) | `28_g2_theory_diagnostic.py`, `29_g2_theory_validation.py` |
| Bundled comparison of shared and fold-restricted encoders (Section S6.4, Table S12) | `05_meta_train.py`, `10_loeo_year_foldwise.py`, `23_reproducibility_multiseed.py`, `06_loeo_kshot.py`, `12_loeo_ablate_alpha.py` |
| Budget-matched control (Sections S2.8 and S5.7, Table S7) | `44_budget_matched_control.py` |
| Source-centroid fusion, fusion-weight scan, facet averaging (Section S3, Fig. S1) | `12_loeo_ablate_alpha.py`, `17_alpha_sensitivity.py`, `15_face_ablation_50reps.py` |
| Negative controls (Section S2.7) | `24negctrl.py` |
| Label-noise curves and confident learning (Section S4) | `40_label_noise_robustness.py`, `41_confident_learning_audit.py` |
| Within-year band attribution (Section 3.5, Section S6.5, Table S13) | `70_band_attribution.py` |
| Random split within a year against the cross-year design, with a rebuild of the published pipeline (Section 3.6, Section S9.1, Table S20) | `81_within_year_split.py` |
| Scaling, wavelength selection and reuse of the selection score (Section 3.4, Section S9.2, Table S21) | `82_chemometric_paths.py` |
| Pooled baselines with labelled held-out samples in the training set (Section 3.4, Section S9.3, Table S22) | `83_symmetric_exposure.py` (iteration cap of the pooled support vector machine: `93_svm_cap_sensitivity.py`) |
| Checkpoint and encoder hyperparameters selected on the held-out domain (Sections 3.2 and 3.4, Section S9.4, Table S23) | `85_checkpoint_paired.py`, `86_encoder_hparam_exposure.py` |
| Exposure contrast against the number of reference samples (Section 3.6, Section S9.5, Table S24) | `87_kcurve_occlusion.py` |
| Band occlusion of the source-only and exposed encoders (Section 3.5, Section S9.6, Table S25) | `87_kcurve_occlusion.py` |
| Supervised training of the encoder and unlabelled exposure on apple and kiwifruit (Sections 3.2 and 3.3, Section S9.7, Table S26) | `88_supervised_unlabelled.py` |
| Robustness of the inference across data sets and seeds (Section 3.2, Section S9.8, Table S27) | `84_inference_robustness.py` |
| Repeated half splits and the unit of the half split (Section 3.3, Section S9.9, Tables S28 and S29) | `89_repeated_half_split.py`, `91_mango_half_unit.py` (counts of the mango table: `94_mango_data_facts.py`) |
| A second origin task: growing region of mango (Section 3.6, Section S9.10, Table S30) | `90_mango_region_task.py` (counts of the mango table: `94_mango_data_facts.py`) |
| Per-class recalls, calibration, facet and anchor permutations, SNV idempotence | `19_perclass_f1_foldwise.py`, `26calibration.py`, `39_face_instrument_confound.py`, `43_snv_idempotency_audit.py` |
| Aggregation of the five-seed formal runs | `28aggregate_multiseed.py` |
| Figures of the main text and Figs. S1–S3 | `67saa_figures.py` |
| Supplementary tables and the number macros printed in the manuscript | `68si_tables.py`, `77_generality_tables.py`, `92_sx_tables.py` |

Shared modules: `protonet.py` (CNN encoder and prototypical network), `encoders.py` (the six encoder
families), `export_utils.py` (paths,
seeds, statistics, result export), `crossdataset_lib.py` and `foundation_lib.py` (public-dataset
experiments). Tests are in `code/tests/`.

`68si_tables.py` and `77_generality_tables.py` write table bodies and number macros into the LaTeX
sources of the manuscript, which are not part of this repository; they check the qualitative
statements of the manuscript against the per-seed results and stop if one does not hold.
`92_sx_tables.py` does the same for Tables S20–S30 and the numbers of the Section S9 analyses, from the
summary workbooks of scripts 81–91 and 93. `67saa_figures.py` runs a
panel-alignment check from an external figure-QA toolkit (`audit_panel_alignment.py`, not
redistributed here); put that module on `PYTHONPATH` before running the script.

## Data availability

The apple spectra are **restricted-access** (internal data of Xinjiang Agricultural University) and
are **not** in this repository; they are described by the data cards in `docs/`, with SHA-256
checksums of every raw archive, and are available from the corresponding author on reasonable
request. The five public datasets are downloadable from their official sources (`data/DATA.md`,
`code/fetch_datasets.py --list`).

## Environment

Python 3.13; `pip install -r requirements.txt` (versions of the reported runs). Encoders were
trained on one NVIDIA RTX 4090 (CUDA) with five fixed seeds (19810915, 20041210, 20060515, 2023,
2024); the spectral-space routes are scikit-learn implementations run on CPU with the same seeds.

## License

Code under MIT (see `LICENSE`). Each external dataset keeps its own license; cite the original
sources.
