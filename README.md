# PCBC: Hard-Boundary Hierarchical Serial Re-centering for Auditable Credit Scoring

## 1. Title

Code and data for the manuscript *"Hard-boundary hierarchical serial re-centering for auditable credit scoring"* (Wei Wang and Li Luo, Business School, Guilin Tourism University), submitted to **PeerJ Computer Science** as an **AI Application** article.

**Permanent archive:** https://doi.org/10.5281/zenodo.24035829

## 2. Description

PCBC is a hierarchical serial re-centering architecture for credit scoring. Each layer re-screens a two-dimensional subspace, builds a Delaunay triangulation of the minority-dense region, and applies a hard-boundary barycentric correction to the residual left by the previous layer. The correction is confined to the convex hull of the training projection, so outside the training support the model falls back to the logistic baseline instead of extrapolating. The single-layer variant, PCBC-1, uses the same geometric core without the serial chain and serves as the ablation baseline.

This package contains the nine preprocessed datasets used in the paper and the self-contained implementation of the logistic base scorer together with the PCBC-1 and PCBC learners, driven by driver scripts that regenerate the reported tables and figures: the multi-baseline comparison of Table 2, the PCBC-1 ablation of Table 3, and the diagnostic analyses behind the remaining tables.

## 3. Dataset Information

All nine datasets are third-party collections redistributed by public repositories. None of the records was collected by the authors; the data are used as distributed, subject to the preprocessing described in Section 7. The nine benchmark matrices in `datasets/` are the preprocessed numeric files actually analysed (last column = binary label, remaining columns = features, median-imputed during curation); the four earlier annual files of the Polish collection, described after the table, support the rolling-window analyses of Table 10.

| # | File in `datasets/` | Dataset name | Source | DOI / URL | N | Features | Minority ratio |
|---|---|---|---|---|---|---|---|
| 1 | `australian_credit.csv` | Statlog (Australian Credit Approval) | UCI ML Repository | 10.24432/C59012 | 690 | 14 | 0.4449 |
| 2 | `japanese_credit.csv` | Credit Approval (Japanese credit screening) | UCI ML Repository | 10.24432/C5FS30 | 690 | 15 | 0.4449 |
| 3 | `german.csv` | Statlog (German Credit Data) | UCI ML Repository | 10.24432/C5NC77 | 1,000 | 20 | 0.3000 |
| 4 | `polish_Bankruptcy5.csv` | Polish Companies Bankruptcy (5th year) | UCI ML Repository | 10.24432/C5F600 | 5,910 | 64 | 0.0694 |
| 5 | `taiwan_bankruptcy.csv` | Taiwanese Bankruptcy Prediction | UCI ML Repository | 10.24432/C5004D | 6,819 | 95 | 0.0323 |
| 6 | `taiwan_credit.csv` | Default of Credit Card Clients | UCI ML Repository | 10.24432/C55S3H | 30,000 | 23 | 0.2212 |
| 7 | `bank-full.csv` | Bank Marketing (bank-full) | UCI ML Repository | 10.24432/C5K306 | 45,211 | 16 | 0.1170 |
| 8 | `US_Bankruptcy.csv` | US Company Bankruptcy Prediction Dataset | Kaggle | https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset (file `american_bankruptcy.csv`, CC0 licence) | 78,682 | 18 | 0.0663 |
| 9 | `give me some credit.csv` | Give Me Some Credit | Kaggle competition | https://www.kaggle.com/c/GiveMeSomeCredit | 150,000 | 10 | 0.0668 |

UCI collections are distributed under CC BY 4.0. The US Company Bankruptcy Prediction Dataset is released under a CC0 1.0 public-domain dedication, and Give Me Some Credit remains subject to the competition terms stated on its Kaggle page.

Feature columns are integer-coded and carry generic placeholder names (`A1`–`A14` for the Australian series, `Attr1`–`AttrN` for the Taiwanese and Polish series, `X1`–`X18` for the US series, and indices `0`–`19` for the German series). Only the last column, named `class`, carries the binary label.

The Polish collection is a five-year series. `polish_Bankruptcy5.csv` (fifth year) is both the Polish entry of the nine-dataset benchmark and the test window of the last rolling task, and the four earlier annual files are redistributed next to it so that the rolling-window protocol of Table 10 can be re-run without any download:

| File | Fiscal year | Rows | Features | Minority ratio |
|---|---|---|---|---|
| `polish_Bankruptcy1.csv` | Y1 | 7,027 | 64 | 0.0386 |
| `polish_Bankruptcy2.csv` | Y2 | 10,173 | 64 | 0.0393 |
| `polish_Bankruptcy3.csv` | Y3 | 10,503 | 64 | 0.0471 |
| `polish_Bankruptcy4.csv` | Y4 | 9,792 | 64 | 0.0526 |
| `polish_Bankruptcy5.csv` | Y5 | 5,910 | 64 | 0.0694 |

These are the same collection, same attributes and same preprocessing as the benchmark matrix; the minority ratios of Y2–Y5 are the ones reported in the note to Table 10 of the manuscript.

## 4. Code Information

### 4.1 Package layout

```
README.md                                  # this file
datasets/                                  # the nine benchmark matrices (Table 1)
                                           #   plus polish_Bankruptcy1..4.csv (Table 10)

code/pcbc_msh3_full9.py                    # PCBC-1 versus PCBC, Table 3 (ablation)
code/pcbc_allmodels_full9.py               # six-model comparison, Table 2 (data source)
code/pcbc_dt_pltr_full9.py                 # decision-tree and StrictPLTR rows of Table 2

code/pcbc_residual_decay_verify.py         # Table 4 (residual decay along the chain)
code/pcbc_ksensitivity.py                  # Table 5 (K-layer sensitivity)
code/pcbc_alpha_sensitivity.py             # Table 6 (fusion-weight sensitivity)
code/pcbc_calibration_diagnostic.py        # Table 7 (ECE and Brier score)
code/pcbc_lr_platt_full9_calibration.py    # Table 7 (LR + Platt calibration reference)
code/pcbc_cost_frontier_full9.py           # Table 8 (audit-cost frontier)
code/pcbc_subgroup_audit.py                # Table 9 (subgroup-level audit)
code/pcbc_audit_trail_case.py              # Table 12 (instance-level audit trail)
code/pcbc_runtime_full9.py                 # Table 13 (per-query inference time)
code/pcbc_sign_condition_full9.py          # sign-condition rate of Lemma 1
code/pcbc_nogate.py                        # Table 11 (hard boundary: gated versus ungated)
code/pcbc_p0_serial_vs_parallel.py         # serial versus parallel head-to-head
code/pcbc_fixedrecall_precision_full9.py   # precision at fixed recall
code/pcbc_temporal_msh3.py                 # Table 10 (rolling window: PCBC-1 versus PCBC)
code/pcbc_temporal_direction1.py           # SHARED MODULE: data loading and base learners;
                                          #   also runs the time-adaptive variants of the
                                          #   rolling-window protocol
code/pcbc_multisubspace_experiment.py      # SHARED MODULE: multi-subspace variant
code/pcbc_multisubspace_static_h3.py       # static-year helper for the multi-subspace variant
code/make_figures.py                       # Figures 1-3
```

`pcbc_msh3_full9.py` is fully standalone. The other entry points load the two shared
modules above by file path (see Section 4.3), so those two files must stay beside the
scripts that use them.

### 4.2 Table and figure to script map

| Manuscript item | Regenerated by |
|---|---|
| Table 1 (dataset summary) | `datasets/` (inspection only) |
| Table 2 (six-model comparison) | `pcbc_allmodels_full9.py`, with `pcbc_dt_pltr_full9.py` |
| Table 3 (PCBC-1 versus PCBC) | `pcbc_msh3_full9.py` |
| Table 4 (residual decay) | `pcbc_residual_decay_verify.py` |
| Table 5 (K sensitivity) | `pcbc_ksensitivity.py` |
| Table 6 (fusion-weight sensitivity) | `pcbc_alpha_sensitivity.py` |
| Table 7 (ECE, Brier score) | `pcbc_calibration_diagnostic.py`, `pcbc_lr_platt_full9_calibration.py` |
| Table 8 (audit cost) | `pcbc_cost_frontier_full9.py` |
| Table 9 (subgroup audit) | `pcbc_subgroup_audit.py` |
| Table 10 (temporal rolling window) | `pcbc_temporal_msh3.py` (PCBC-1 versus PCBC, Tasks 1–4); `pcbc_temporal_direction1.py` re-runs the same protocol with the time-adaptive variants |
| Table 11 (hard-boundary shift) | `pcbc_nogate.py` (gated versus ungated, shift sweep on Train Y1 → Test Y5) |
| Table 12 (audit trail case) | `pcbc_audit_trail_case.py` |
| Table 13 (inference time) | `pcbc_runtime_full9.py` |
| Figures 1-3 | `make_figures.py` |

### 4.3 Implementation notes

The scripts are standalone entry points and can be run directly from the package root or from inside `code/`. Each script locates the package root as `dirname(dirname(abspath(__file__)))`, reads its inputs from `datasets/` and writes its result CSV next to `README.md`, so the package can be moved to another machine unchanged. No installation step, `PYTHONPATH` setting or package import is required.

Twenty-one entry points are shipped. Two scripts also act as shared modules: `pcbc_temporal_direction1.py` holds the data loader and the base learners, and `pcbc_multisubspace_experiment.py` holds the multi-subspace variant. The other entry points load them by file path with `importlib`, which is why the package needs no installed module and why those two files must be kept in `code/`. `pcbc_msh3_full9.py` is fully standalone; `pcbc_allmodels_full9.py` additionally loads `pcbc_temporal_direction1.py`.

The implementation relies only on open-source primitives (SciPy Qhull for the Delaunay triangulation, scikit-learn for the base learner, cross-validation and metrics); no third-party code is vendored, and the interpretable additive baseline (EBM) is implemented in this package rather than taken from an external interpretability toolkit. All comments, docstrings and printed messages are in English.

The permanent, citable copy of this package is archived at Zenodo: https://doi.org/10.5281/zenodo.24035829

## 5. Usage Instructions

```
python code/pcbc_msh3_full9.py       # Table 3, from the package root
python code/pcbc_allmodels_full9.py  # Table 2, from the package root
# or
cd code && python pcbc_msh3_full9.py
```

`pcbc_msh3_full9.py` loops over the nine datasets, runs the two models under the paired five-fold protocol, prints the per-dataset summary and writes `pcbc_msh3_full9_results.csv` to the package root. One row is a dataset × model summary (mean ± std over the five folds) of Precision / Recall / F1 / AUC / PR-AUC, i.e. the PCBC-1 versus PCBC rows of Table 3. The full run takes a few minutes per dataset on a laptop-class CPU (five folds × two models); a single fold fit takes well under a minute.

`pcbc_allmodels_full9.py` reads `pcbc_msh3_full9_results.csv`, so `pcbc_msh3_full9.py` has to be run first; every other entry point is independent and writes its own result file to the package root, and the mapping is given in Section 4.2. The three rolling-window scripts (`pcbc_temporal_msh3.py`, `pcbc_temporal_direction1.py`, `pcbc_nogate.py`) read `polish_Bankruptcy1.csv`–`polish_Bankruptcy5.csv`. A run of `pcbc_temporal_msh3.py` takes a few minutes for the early tasks and longer for the last one, whose training window spans four years.

## 6. Requirements

- Python 3.10 or later (verified with Python 3.13.16)
- numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1
- xgboost (optional; the XGBoost baseline rows of Table 2 are skipped when it is unavailable)
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

Wang W, Luo L. Hard-boundary hierarchical serial re-centering for auditable credit scoring. PeerJ Computer Science.

and the archived code and data package:

Wang W, Luo L. PCBC: hard-boundary hierarchical serial re-centering for auditable credit scoring, reproduction package. Zenodo. https://doi.org/10.5281/zenodo.24035829

Please also cite the original dataset providers:

- Quinlan R. Statlog (Australian Credit Approval). UCI Machine Learning Repository. doi:10.24432/C59012
- Credit Approval. UCI Machine Learning Repository. doi:10.24432/C5FS30
- Hofmann H. Statlog (German Credit Data). UCI Machine Learning Repository. doi:10.24432/C5NC77
- Tomczak S. Polish Companies Bankruptcy. UCI Machine Learning Repository. doi:10.24432/C5F600
- Taiwanese Bankruptcy Prediction. UCI Machine Learning Repository. doi:10.24432/C5004D
- Yeh I. Default of Credit Card Clients. UCI Machine Learning Repository. doi:10.24432/C55S3H
- Moro S, Rita P, Cortez P. Bank Marketing. UCI Machine Learning Repository. doi:10.24432/C5K306
- Singh U. US Company Bankruptcy Prediction Dataset. Kaggle. https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset
- Give Me Some Credit. Kaggle. https://www.kaggle.com/c/GiveMeSomeCredit

## 9. Licence and Contribution Guidelines

The code in this package is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0), the licence under which the archived copy is deposited at Zenodo. The datasets are not relicensed here: the UCI collections remain under CC BY 4.0, and the two Kaggle collections remain under the terms stated on their dataset pages.

Corrections and extensions are welcome: open an issue or a pull request on the archived repository, or write to the corresponding author (Li Luo, Business School, Guilin Tourism University). Re-running a script and reporting a mismatch against the manuscript is the most useful contribution at this stage.
