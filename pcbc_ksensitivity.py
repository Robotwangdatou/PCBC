# -*- coding: utf-8 -*-
"""
K-layer Sensitivity of Serial Re-centering (PCBC, hier mode)
============================================================
Purpose: justify the choice K=3 - run K=1/2/3/4 on representative datasets and compare the full
      performance, checking whether P/R/F1/AUC/PR-AUC saturates beyond K=3 (no further real gain).

Same protocol and construction as the main comparison of the manuscript (Table 3, pcbc_msh3_full9.py):
  - model: serial layer-by-layer re-centering of exp4.PCBC_MS (mode='hier', re-selecting the single
          best pair for that layer's residual at every layer, use_bary=True, alpha=0.6, seed=42)
  - protocol: StratifiedKFold(5, shuffle=True, random_state=42), paired five folds
  - preprocessing: z-scoring with training-fold statistics (no leakage)
  - TOP_K: clipped by sample size (n<30k->30, 30k-100k->20, >=100k->12)
  - metrics: Precision / Recall / F1 / AUC / PR-AUC (minority class positive, Table 5 convention)

Key implementation detail: cumulative layer by layer. The serial-chain logit is LR + alpha * sum_{i<=k} sim_i, so
the first layer of K=2 equals the first layer of K=3, and the first three layers of K=4 equal all layers of K=3.
Each fold screens pairs and fits once per layer, and the K=1..4 predictions share the same lower layers,
so the comparison is clean (strictly fold-aligned) and saves 5*(1+2+3+4)-5*4 = 30 redundant screenings.

Run:
  python pcbc_ksensitivity.py
Output:
  pcbc_ksensitivity_agg.csv   (dataset x K aggregate: mean +/- std)
  pcbc_ksensitivity_fold.csv  (fold-level detail)
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
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

ALPHA = getattr(exp1, "ALPHA", 0.6)
N_FOLDS = 5
SEED = 42
K_MAX = 4                     # sensitivity sweep K=1..4 (K=3 is the configuration used in the paper)
DATASETS = ["polish_Bankruptcy5",   # typical imbalanced case (0.0694), the temporal main case of the paper
                        "german"]                # typical balanced case (0.3000)


def effective_topk(n):
    if n >= 100000:
        return 12
    if n >= 30000:
        return 20
    return 30


# ------------------------- cumulative layer-by-layer serial PCBC -------------------------
class PCBC_HierK(BaseEstimator, ClassifierMixin):
    """
    Structurally identical to exp4.PCBC_MS(mode='hier'), except that all layers are kept,
    so that one fit yields the decision function of every K=1..K_MAX (shared lower layers).
    """
    def __init__(self, alpha=ALPHA, seed=SEED, use_bary=True, K_max=K_MAX):
        self.alpha = alpha
        self.seed = seed
        self.use_bary = use_bary
        self.K_max = K_max
        self.lr = None
        self.layers = []

    def fit(self, X, y, year_id=None):
        if year_id is None:
            year_id = np.ones(len(y))
        self.lr = LogisticRegression(max_iter=500, class_weight="balanced",
                                     random_state=self.seed)
        self.lr.fit(X, y)
        prob = self.lr.predict_proba(X)[:, 1]
        self.layers = []
        for k in range(self.K_max):
            resid_k = y - prob
            # re-select the single best pair for that layer's residual (same call as exp4.PCBC_MS.hier)
            pair_k, _ = exp4.select_top_k_pairs(
                X, y, self.lr.coef_[0], top_k=exp4.TOP_K, K=1,
                seed=self.seed, target_resid=resid_k)
            sim = exp1.TemporalLSIM(pair_k[0], use_bary=self.use_bary,
                                    seed=self.seed)
            sim.fit(X, resid_k, y, year_id)
            self.layers.append(sim)
            # update the current cumulative probability (layerwise re-centering); the next layer fits the remaining residual
            logit = self.lr.decision_function(X) + self.alpha * sim.predict(X)
            prob = exp1.logit_to_prob(logit)
        return self

    def decision_function(self, X, K):
        logit = self.lr.decision_function(X)
        for sim in self.layers[:K]:
            logit += self.alpha * sim.predict(X)
        return logit

    def predict_proba(self, X, K):
        p = exp1.logit_to_prob(self.decision_function(X, K))
        return np.vstack([1 - p, p]).T


# ------------------------- single-dataset evaluation (paired five folds, K=1..4 fold-aligned) -------------------------
def eval_dataset(ds_name):
    X, y, minority_label, _ = exp1.load_dataset(ds_name)
    exp4.TOP_K = effective_topk(len(y))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        m = PCBC_HierK(seed=SEED)
        m.fit(Xtr, y[tr])
        uniq = np.unique(y[te])
        idx_min = 0 if uniq[0] == minority_label else 1
        for K in range(1, K_MAX + 1):
            prob = m.predict_proba(Xte, K)[:, 1]
            yp = (prob >= 0.5).astype(int)
            prf = precision_recall_fscore_support(y[te], yp, average=None,
                                                  zero_division=0)
            rows.append({
                "dataset": ds_name, "K": K, "fold": fold,
                "P": prf[0][idx_min], "R": prf[1][idx_min],
                "F1": prf[2][idx_min], "AUC": roc_auc_score(y[te], prob),
                "PR_AUC": average_precision_score(y[te], prob),
            })
        print(f"  [fold {fold}] {ds_name} done", flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    all_fold = pd.concat([eval_dataset(ds) for ds in DATASETS],
                         ignore_index=True)
    agg = all_fold.groupby(["dataset", "K"])[["P", "R", "F1", "AUC",
                                              "PR_AUC"]].agg(
        ["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]

    for ds in DATASETS:
        sub = agg.loc[ds]
        print(f"\n===== K-layer sensitivity: {ds} =====", flush=True)
        print(sub.to_string(), flush=True)
        # increment relative to K=3 (the paper configuration), which shows whether performance saturates
        base = sub.loc[3]
        deltas = sub[["F1_mean", "AUC_mean", "PR_AUC_mean"]].sub(base[
            ["F1_mean", "AUC_mean", "PR_AUC_mean"]])
        print("delta vs K=3 (F1 / AUC / PR-AUC):", flush=True)
        print(deltas.round(4).to_string(), flush=True)

    # K means merged over the two datasets (cross-dataset average, same scale as the paper)
    print("\n===== mean over datasets by K (P / R / F1 / AUC / PR-AUC) =====",
          flush=True)
    grand = all_fold.groupby("K")[["P", "R", "F1", "AUC",
                                   "PR_AUC"]].mean().round(4)
    print(grand.to_string(), flush=True)

    agg.to_csv(os.path.join(_REPO, "pcbc_ksensitivity_agg.csv"))
    all_fold.to_csv(os.path.join(_REPO, "pcbc_ksensitivity_fold.csv"),
                    index=False)
    print(f"\n[Saved] pcbc_ksensitivity_agg.csv + pcbc_ksensitivity_fold.csv "
          f"| total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
