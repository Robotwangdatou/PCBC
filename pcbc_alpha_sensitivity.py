# -*- coding: utf-8 -*-
"""
Fusion-Weight (alpha) Sensitivity of PCBC
=========================================
Purpose: justify the fixed alpha=0.6 - sweep alpha in {0.2,0.4,0.6,0.8,1.0} and check whether
      P/R/F1/AUC/PR-AUC change with the fusion weight, confirming that performance near 0.6 is flat
      (robust) rather than a sharp peak (fragile). alpha=0 is plain LR (its numbers are already in Table 3, not re-run).

Same protocol and construction as the main comparison of the manuscript (Table 3, pcbc_msh3_full9.py):
  - model: exp4.PCBC_MS(mode='hier', K=3, use_bary=True) - serial three-layer re-centering
    (re-selecting the single best pair for that layer's residual at every layer), i.e. PCBC; alpha is a native model parameter
  - protocol: StratifiedKFold(5, shuffle=True, random_state=42), paired five folds
  - preprocessing: z-scoring with training-fold statistics (no leakage)
  - TOP_K: clipped by sample size (n<30k->30, 30k-100k->20, >=100k->12)
  - metrics: Precision / Recall / F1 / AUC / PR-AUC (minority class positive, Table 6 convention)

Why every alpha is re-run in full (unlike the cumulative K sensitivity):
  the residual of layer k>1 is y - prob_{k-1}, and prob_{k-1} already contains an alpha * sim term,
  so alpha changes the candidate residual, the pair selection and the interpolation fit of the later
  layers - the training procedure itself depends on alpha, so changing only the inference weight is not enough.

Run:
  python pcbc_alpha_sensitivity.py
Output:
  pcbc_alpha_sensitivity_agg.csv   (dataset x alpha aggregate: mean +/- std)
  pcbc_alpha_sensitivity_fold.csv  (fold-level detail)
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
import importlib.util

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BASE)
_spec1 = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_BASE, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec1)
_spec1.loader.exec_module(exp1)
_spec4 = importlib.util.spec_from_file_location(
    "exp4", os.path.join(_BASE, "pcbc_multisubspace_experiment.py"))
exp4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(exp4)

N_FOLDS = 5
SEED = 42
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]      # 0.6 is the configuration used in the paper
DATASETS = ["polish_Bankruptcy5",        # typical imbalanced case (0.0694), the temporal main case of the paper
                        "german"]                     # typical balanced case (0.3000)


def effective_topk(n):
    if n >= 100000:
        return 12
    if n >= 30000:
        return 20
    return 30


# ------------------------- single-dataset evaluation (paired five folds, each alpha fitted in full) -------------------------
def eval_dataset(ds_name):
    X, y, minority_label, _ = exp1.load_dataset(ds_name)
    exp4.TOP_K = effective_topk(len(y))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        uniq = np.unique(y[te])
        idx_min = 0 if uniq[0] == minority_label else 1
        for a in ALPHAS:
            # same construction as the paper: three serial layers, use_bary=True, alpha as a native parameter
            m = exp4.PCBC_MS(seed=SEED, K=3, mode="hier", use_bary=True,
                             alpha=a)
            m.fit(Xtr, y[tr])
            prob = m.predict_proba(Xte)[:, 1]
            yp = (prob >= 0.5).astype(int)
            prf = precision_recall_fscore_support(y[te], yp, average=None,
                                                  zero_division=0)
            rows.append({
                "dataset": ds_name, "alpha": a, "fold": fold,
                "P": prf[0][idx_min], "R": prf[1][idx_min],
                "F1": prf[2][idx_min], "AUC": roc_auc_score(y[te], prob),
                "PR_AUC": average_precision_score(y[te], prob),
            })
        print(f"  [fold {fold}] {ds_name} done (alpha 0.2..1.0)", flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    all_fold = pd.concat([eval_dataset(ds) for ds in DATASETS],
                         ignore_index=True)
    agg = all_fold.groupby(["dataset", "alpha"])[["P", "R", "F1", "AUC",
                                                  "PR_AUC"]].agg(
        ["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]

    for ds in DATASETS:
        sub = agg.loc[ds]
        print(f"\n===== alpha sensitivity: {ds} =====", flush=True)
        print(sub.to_string(), flush=True)
        # increment relative to alpha=0.6 (the paper configuration), which shows whether the curve is flat
        base = sub.loc[0.6]
        deltas = sub[["F1_mean", "AUC_mean", "PR_AUC_mean"]].sub(base[
            ["F1_mean", "AUC_mean", "PR_AUC_mean"]])
        print("delta vs alpha=0.6 (F1 / AUC / PR-AUC):", flush=True)
        print(deltas.round(4).to_string(), flush=True)

    print("\n===== mean over datasets by alpha (P / R / F1 / AUC / PR-AUC) =====",
          flush=True)
    grand = all_fold.groupby("alpha")[["P", "R", "F1", "AUC",
                                       "PR_AUC"]].mean().round(4)
    print(grand.to_string(), flush=True)

    agg.to_csv(os.path.join(_REPO, "pcbc_alpha_sensitivity_agg.csv"))
    all_fold.to_csv(os.path.join(_REPO, "pcbc_alpha_sensitivity_fold.csv"),
                    index=False)
    print(f"\n[Saved] pcbc_alpha_sensitivity_agg.csv + "
          f"pcbc_alpha_sensitivity_fold.csv | total {time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
