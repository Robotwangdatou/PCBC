# PCBC: Hard-Boundary Hierarchical Serial Re-centering for Auditable Credit Scoring

Reproduction package for the main experiment (Table III) of *"Hard-Boundary
Hierarchical Serial Re-centering for Auditable Credit Scoring"* (IEEE Access
submission).

## Repository layout

```
上传仓库/
├── datasets/                    # the nine datasets used in the paper (Table I)
└── code/
    └── pcbc_msh3_full9.py       # self-contained main experiment (Table III)
```

`pcbc_msh3_full9.py` is **fully self-contained**: it defines the data loader,
the Delaunay residual-correction learner and both models (PCBC-1 single-layer
and PCBC hierarchical serial) in one file. No other project modules are
required; the only runtime dependencies are `numpy`, `pandas`, `scipy` and
`scikit-learn`. The script locates the datasets automatically under
`../datasets/` relative to itself, so the package can be moved anywhere.

## Datasets (Table I)

| # | File (in `datasets/`) | Source |
|---|----------------------|--------|
| 1 | `australian_credit.csv` | UCI Statlog (Australian Credit) |
| 2 | `german.csv` | UCI Statlog (German Credit) |
| 3 | `taiwan_credit.csv` | UCI Taiwanese Credit |
| 4 | `taiwan_bankruptcy.csv` | Taiwanese Bankruptcy Prediction |
| 5 | `polish_Bankruptcy5.csv` | UCI Polish Companies Bankruptcy (5th year) |
| 6 | `US_Bankruptcy.csv` | Kaggle US Bankruptcy |
| 7 | `bank-full.csv` | UCI Bank Marketing |
| 8 | `japanese_credit.csv` | UCI Japanese Credit |
| 9 | `give me some credit.csv` | Kaggle Give Me Some Credit |

## Environment

- Python 3.8+
- `numpy`, `pandas`, `scipy`, `scikit-learn`

## Reproducing Table III

```
cd code
python pcbc_msh3_full9.py
```

This runs the main experiment on all nine datasets (about 3–5 minutes per
dataset) and writes `pcbc_msh3_full9_results.csv` to the repository root. Each
row is a dataset × model summary (mean ± std over the five folds) of
Precision / Recall / F1 / AUC / PR-AUC, exactly the layout of Table III
(Single-Layer PCBC-1 versus PCBC).

## Protocol (identical to the paper)

- Label = last CSV column; the minority class is the positive class.
- Paired stratified 5-fold cross-validation:
  `StratifiedKFold(5, shuffle=True, random_state=42)`.
- Train-only z-score normalisation; decision threshold 0.5.
- `alpha = 0.6` (fusion weight of the geometric correction).
- Candidate-pair budget by sample size: `n >= 100000 -> 12`,
  `n >= 30000 -> 20`, otherwise `30`.
- Models: `PCBC-1` (K=1, single layer, ablation baseline) and
  `PCBC` (K=3, hierarchical serial chain).
- Random seed fixed at 42; the paper numbers correspond to this seed.
