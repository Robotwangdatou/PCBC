# PCBC: Hard-Boundary Hierarchical Serial Re-centering for Auditable Credit Scoring

## 1. Title

Code and data for the manuscript *"Hard-boundary hierarchical serial re-centering for auditable credit scoring"* (Wei Wang and Li Luo, Business School, Guilin Tourism University), submitted to **PeerJ Computer Science** as an **AI Application** article (manuscript in review).

## 2. Description

PCBC is a hierarchical serial re-centering architecture for credit scoring. Each layer re-screens a two-dimensional subspace, builds a Delaunay triangulation of the minority-dense region, and applies a hard-boundary barycentric correction to the residual left by the previous layer. The correction is confined to the convex hull of the training projection, so outside the training support the model falls back to the logistic baseline instead of extrapolating. The single-layer variant, PCBC-1, uses the same geometric core without the serial chain and serves as the ablation baseline.

This repository contains the nine preprocessed datasets used in the paper and the self-contained script that reproduces the main nine-dataset comparison (Table 3 of the manuscript).

## 3. Dataset Information

All nine datasets are third-party collections redistributed by public repositories. None of the records was collected by the authors; the data are used as distributed, subject to the preprocessing described in Section 6. The files in `datasets/` are the preprocessed numeric matrices actually analysed (last column = binary label, remaining columns = features, median-imputed and scaled during curation).

| # | File in `datasets/` | Dataset name | Source | DOI / URL | N | Features | Minority ratio |
|---|---|---|---|---|---|---|---|
| 1 | `australian_credit.csv` | Statlog (Australian Credit Approval) | UCI ML Repository | 10.24432/C59012 | 690 | 14 | 0.4449 |
| 2 | `japanese_credit.csv` | Credit Approval (Japanese credit screening) | UCI ML Repository | 10.24432/C5FS30 | 690 | 15 | 0.4449 |
| 3 | `german.csv` | Statlog (German Credit Data) | UCI ML Repository | 10.24432/C5NC77 | 1,000 | 20 | 0.3000 |
| 4 | `polish_Bankruptcy5.csv` | Polish Companies Bankruptcy (5th year) | UCI ML Repository | 10.24432/C5F600 | 5,910 | 64 | 0.0694 |
| 5 | `taiwan_bankruptcy.csv` | Taiwanese Bankruptcy Prediction | UCI ML Repository | 10.24432/C5004D | 6,819 | 95 | 0.0323 |
| 6 | `taiwan_credit.csv` | Default of Credit Card Clients | UCI ML Repository | 10.24432/C55S3H | 30,000 | 23 | 0.2212 |
| 7 | `bank-full.csv` | Bank Marketing (bank-full) | UCI ML Repository | 10.24432/C5K306 | 45,211 | 16 | 0.1170 |
| 8 | `US_Bankruptcy.csv` | US Company Bankruptcy Prediction Dataset | Kaggle | https://www.kaggle.com/datasets?search=american+bankruptcy (dataset title "US Company Bankruptcy Prediction Dataset", file `american_bankruptcy.csv`) | 78,682 | 18 | 0.0663 |
| 9 | `give me some credit.csv` | Give Me Some Credit | Kaggle competition | https://www.kaggle.com/c/GiveMeSomeCredit | 150,000 | 10 | 0.0668 |

UCI collections are distributed under CC BY 4.0; the two Kaggle collections remain subject to the terms stated on their dataset pages.

## 4. Code Information

```
code/pcbc_msh3_full9.py                  # PCBC-1 versus PCBC on nine datasets (Table 3)
code/pcbc_baselines_full9.py             # external baselines, five baselines (Table 2)
code/pcbc_dt_pltr_full9.py               # decision tree and StrictPLTR baselines (Table 2)
code/pcbc_allmodels_full9.py             # combined multi-model comparison (Table 2)
code/pcbc_residual_decay_verify.py       # layer-wise residual norms (Table 4)
code/pcbc_p0_serial_vs_parallel.py       # serial chain versus parallel construction (Table 4)
code/pcbc_ksensitivity.py                # layer-count sensitivity, K = 1..4 (Table 5)
code/pcbc_alpha_sensitivity.py           # fusion-weight sensitivity, alpha (Table 6)
code/pcbc_calibration_diagnostic.py      # ECE and Brier score (Table 7)
code/pcbc_lr_platt_full9_calibration.py  # LR + Platt calibration reference (Table 7)
code/pcbc_cost_frontier_full9.py         # normalized audit cost by cost ratio (Table 8)
code/pcbc_subgroup_audit.py              # subgroup audit (Table 9)
code/pcbc_temporal_direction1.py         # rolling-window temporal benchmark (Table 10)
code/pcbc_multisubspace_static_h3.py     # hard-boundary robustness under shift (Table 11)
code/pcbc_multisubspace_experiment.py    # shared multi-subspace experiment driver
code/pcbc_nogate.py                      # gate ablation supporting Table 11
code/pcbc_audit_trail_case.py            # serial audit trail of one record (Table 12)
code/pcbc_runtime_full9.py               # per-query inference time (Table 13)
code/pcbc_fixedrecall_precision_full9.py # fixed-Recall precision analysis (Section 4)
code/pcbc_sign_condition_full9.py        # sign-condition verification (Section 3.4)
code/make_figures.py                     # Figures 1-3
datasets/                                # the nine preprocessed datasets (Table 1)
README.md                                # this file
```

Every script is an entry point and can be run directly from the package root. Each one locates the package root as `dirname(dirname(abspath(__file__)))`, reads its inputs from `datasets/` and writes its result CSV next to `README.md`, so the package can be moved to another machine unchanged. Scripts that need a shared experiment driver load it from the same `code/` directory by file path, so no installation step, `PYTHONPATH` setting or package import is required.

Run order. `code/pcbc_msh3_full9.py` should be run first: it writes `pcbc_msh3_full9_results.csv`, which the baseline and multi-model scripts merge with their own results to guarantee an identical fold assignment. `code/pcbc_cost_frontier_full9.py` and `code/pcbc_residual_decay_verify.py` must precede `code/make_figures.py`, which plots Figure 2 from the cost-frontier summary and Figure 3 from the residual-decay output. The remaining scripts are independent of each other.

The implementation relies only on open-source primitives (SciPy Qhull for the Delaunay triangulation, scikit-learn for the base learner, cross-validation and metrics); no third-party code is vendored. Comments and print statements inside some scripts are in Chinese, as they were carried over from the working repository; all identifiers, function names and output filenames are in English.

### Coverage map: manuscript tables to scripts

| Manuscript table or figure | Script(s) in this package |
|---|---|
| Table 1 | none (compiled from `datasets/`) |
| Table 2 | `pcbc_baselines_full9.py`, `pcbc_dt_pltr_full9.py`, `pcbc_allmodels_full9.py` |
| Table 3 | `pcbc_msh3_full9.py` |
| Table 4 | `pcbc_residual_decay_verify.py`, `pcbc_p0_serial_vs_parallel.py` |
| Table 5 | `pcbc_ksensitivity.py` |
| Table 6 | `pcbc_alpha_sensitivity.py` |
| Table 7 | `pcbc_calibration_diagnostic.py`, `pcbc_lr_platt_full9_calibration.py` |
| Table 8 | `pcbc_cost_frontier_full9.py` |
| Table 9 | `pcbc_subgroup_audit.py` |
| Table 10 | `pcbc_temporal_direction1.py` |
| Table 11 | `pcbc_multisubspace_static_h3.py`, `pcbc_multisubspace_experiment.py`, `pcbc_nogate.py` |
| Table 12 | `pcbc_audit_trail_case.py` |
| Table 13 | `pcbc_runtime_full9.py` |
| Figures 1-3 | `make_figures.py` |
| Text-only analyses | `pcbc_fixedrecall_precision_full9.py`, `pcbc_sign_condition_full9.py` |

## 5. Usage Instructions

```
python code/pcbc_msh3_full9.py       # from the package root
# or
cd code && python pcbc_msh3_full9.py # from inside the code folder
```

The script loops over the nine datasets, runs the two models under the paired five-fold protocol, prints the per-dataset summary and writes `pcbc_msh3_full9_results.csv` to the package root. One row is a dataset × model summary (mean ± std over the five folds) of Precision / Recall / F1 / AUC / PR-AUC, i.e. the layout of Table 3 of the manuscript. The full run takes a few minutes per dataset on a laptop-class CPU (five folds x two models); a single fold fit takes well under a minute.

## 6. Requirements

- Python 3.10 or later (verified with Python 3.13.16)
- numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1
- CPU only; no GPU, no proprietary solver and no external interpretability toolkit is required
- No internet access is needed at run time: the datasets are included in `datasets/`

The environment used for the reported results: Microsoft Windows 11 (Home, build 10.0.26200); Intel Core i7-12700F (12 physical cores, 20 logical threads); 16 GB RAM (15.8 GB usable).

## 7. Methodology

Preprocessing (performed once, before any model fitting):

- The last column of each CSV is the binary label; the minority (default/bankruptcy) class is the positive class.
- Categorical attributes are integer-coded with an ordinal label encoder (one code per level, no dummy expansion).
- Missing entries (the `?` placeholder of the Japanese screening data, and the absent values of Give Me Some Credit and the Polish series) are replaced by the median of the corresponding attribute.
- No resampling, no synthetic minority generation and no class-conditional transformation: imbalance is handled by balanced class weights in the base learners and by the minority-dense screening of the geometric layer.
- The curated series were standardized once (zero mean, unit variance) during curation; the protocol re-standardizes each training fold with its own mean and standard deviation and applies the same transform to the held-out fold.

Protocol (identical to the manuscript):

- Paired stratified 5-fold cross-validation, `StratifiedKFold(5, shuffle=True, random_state=42)`.
- Decision threshold 0.5; fusion weight `alpha = 0.6`.
- Candidate-pair budget by sample size: `n >= 100000 -> 12`, `n >= 30000 -> 20`, otherwise `30`.
- Models: PCBC-1 (K = 1, single layer, ablation baseline) and PCBC (K = 3, hierarchical serial chain).
- Random seed fixed at 42; the reported numbers correspond to this seed.

## 8. Citations

If you use this code or these data, please cite the manuscript:

Wang W, Luo L. Hard-boundary hierarchical serial re-centering for auditable credit scoring. PeerJ Computer Science (manuscript in review).

Please also cite the original dataset providers:

- Quinlan R. Statlog (Australian Credit Approval). UCI Machine Learning Repository. doi:10.24432/C59012
- Credit Approval. UCI Machine Learning Repository. doi:10.24432/C5FS30
- Hofmann H. Statlog (German Credit Data). UCI Machine Learning Repository. doi:10.24432/C5NC77
- Tomczak S. Polish Companies Bankruptcy. UCI Machine Learning Repository. doi:10.24432/C5F600
- Taiwanese Bankruptcy Prediction. UCI Machine Learning Repository. doi:10.24432/C5004D
- Yeh I. Default of Credit Card Clients. UCI Machine Learning Repository. doi:10.24432/C55S3H
- Moro S, Rita P, Cortez P. Bank Marketing. UCI Machine Learning Repository. doi:10.24432/C5K306
- US Company Bankruptcy Prediction Dataset. Kaggle. https://www.kaggle.com/datasets?search=american+bankruptcy
- Give Me Some Credit. Kaggle. https://www.kaggle.com/c/GiveMeSomeCredit

## 9. License and Contribution Guidelines

The code in this package is released under the MIT License. The datasets are not relicensed here: UCI collections remain under CC BY 4.0, and the two Kaggle collections remain under the terms stated on their dataset pages.

Corrections and extensions are welcome: open an issue or a pull request on the archived repository, or write to the corresponding author (Li Luo, Business School, Guilin Tourism University). Re-running a script and reporting a mismatch against the manuscript is the most useful contribution at this stage.
