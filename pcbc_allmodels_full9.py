# -*- coding: utf-8 -*-
"""
Six-model comparison over the nine datasets (data source for Table 2)
====================================================
Same protocol (nine datasets, StratifiedKFold(5, seed=42), training-fold statistics
for z-scoring, threshold 0.5, minority-class P/R/F1/AUC/PR-AUC), row-identical to pcbc_msh3_full9.py.

External baselines (actually run by this script):
  LR        : LogisticRegression(class_weight='balanced', max_iter=500)
    XGBoost   : n_estimators=50, max_depth=3, scale_pos_weight=neg/pos (same configuration as the temporal reference table of the manuscript)
  RF        : RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
    EBM       : hand-written Explainable Boosting (GA2M style) - binned-histogram gradient boosting, no dependency on the interpret package

Own models (not re-fitted: read from pcbc_msh3_full9_results.csv and merged, same folds and seed):
  PCBC, PCBC-MS-H3

Dependencies: numpy / pandas / sklearn (required), xgboost (XGBoost is skipped with a message when missing)
Run: python pcbc_allmodels_full9.py   (run pcbc_msh3_full9.py first to create pcbc_msh3_full9_results.csv)
Output: pcbc_allmodels_full9.csv (one row per dataset and model, mean +/- std aggregation)
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

_spec = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_allmodels_full9.csv")
FOLD_CSV = os.path.join(DATA_DIR, "pcbc_allmodels_full9_fold.csv")
OWN_CSV = os.path.join(DATA_DIR, "pcbc_msh3_full9_results.csv")


DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]


# ---------------- hand-written EBM (GA2M style, no dependency on interpret) ----------------
class EBMHandcrafted:
    """
    Lightweight implementation of the Explainable Boosting Machine:
        - one binned-histogram shape function term_j(x_j) per feature
        - n_rounds outer rounds, each running round-robin gradient boosting over all features
        - gradients weighted by balanced sample weights; the bin mean is the increment of that bin
        - every feature term is re-centred to its own mean each round (identifiability constraint, zero-mean shape functions)
    Prediction: p = sigmoid(base + sum_j term_j(x_j))
    """
    def __init__(self, n_bins=64, n_rounds=20, learning_rate=0.05):
        self.n_bins = n_bins
        self.n_rounds = n_rounds
        self.lr = learning_rate
        self.bin_edges = None   # list of arrays, per feature
        self.terms = None       # list of arrays, per feature
        self.base = 0.0
        self.fit_time = None

    def fit(self, X, y):
        t0 = time.time()
        n, d = X.shape
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        w = np.where(y == 1, 1.0 / max(n1, 1), 1.0 / max(n0, 1))
        w = w / w.mean()                        # balanced, normalised to unit mean

        # bin edges: training-set quantiles (independently per feature)
        self.bin_edges = []
        for j in range(d):
            q = np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1))
            q = np.unique(q)
            if len(q) < 2:
                # constant or near-constant feature (its standard deviation is ~0 after z-scoring, so the quantiles
                # deduplicate to a single point): widened to one wide bin, the feature then contributes zero
                c = float(q[0])
                q = np.array([c - 1.0, c + 1.0])
            self.bin_edges.append(q)
        self.terms = [np.zeros(len(edges) - 1) for edges in self.bin_edges]
        self.base = float(np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6)))

        # pre-bin the training data (avoids re-running digitize every round)
        X_bins = np.column_stack([
            np.clip(np.digitize(X[:, j], self.bin_edges[j]) - 1,
                    0, len(self.terms[j]) - 1)
            for j in range(d)
        ])

        for _ in range(self.n_rounds):
            logit = np.full(n, self.base)
            for j in range(d):
                logit += self.terms[j][X_bins[:, j]]
            p = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
            grad = (y - p) * w
            for j in range(d):
                t = self.terms[j]
                for b in range(len(t)):
                    m = X_bins[:, j] == b
                    if m.any():
                        t[b] += self.lr * grad[m].mean()
                t -= t.mean()                    # identifiability constraint
        self.fit_time = round(time.time() - t0, 1)
        return self

    def _logit(self, X):
        n = len(X)
        logit = np.full(n, self.base)
        for j in range(len(self.terms)):
            nb = len(self.terms[j])
            bins = np.clip(np.digitize(X[:, j], self.bin_edges[j]) - 1,
                           0, nb - 1)
            logit += self.terms[j][bins]
        return logit

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-np.clip(self._logit(X), -50, 50)))
        return np.vstack([1 - p, p]).T


# ---------------- external baseline construction (built at run time, see run_dataset) ----------------
AVAIL = {}


def run_dataset(ds_name):
    t0 = time.time()
    X, y, minority_label, min_ratio = exp1.load_dataset(ds_name)
    n = len(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg

        # LR / RF / EBM
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        models = [
            ("LR", LogisticRegression(class_weight="balanced", max_iter=500)),
            ("RF", RandomForestClassifier(class_weight="balanced",
                                          n_estimators=200, random_state=42, n_jobs=-1)),
            ("EBM", EBMHandcrafted(n_bins=64, n_rounds=20, learning_rate=0.05)),
        ]
        # XGBoost (when available)
        if AVAIL.get("XGBoost"):
            import xgboost
            pos = int((y[tr] == 1).sum()); neg = int((y[tr] == 0).sum())
            models.append(("XGBoost", xgboost.XGBClassifier(
                n_estimators=50, max_depth=3,
                scale_pos_weight=neg / max(pos, 1),
                random_state=42, n_jobs=-1, eval_metric="logloss")))

        for vname, m in models:
            m.fit(Xtr, y[tr])
            prob = m.predict_proba(Xte)[:, 1]
            y_pred = (prob >= 0.5).astype(int)
            prf = precision_recall_fscore_support(y[te], y_pred,
                                                  average=None, zero_division=0)
            uniq = np.unique(y[te])
            idx_min = 0 if uniq[0] == minority_label else 1
            rows.append({
                "dataset": ds_name, "model": vname, "fold": fold,
                "precision": prf[0][idx_min], "recall": prf[1][idx_min],
                "f1": prf[2][idx_min],
                "auc": roc_auc_score(y[te], prob),
                "auc_pr": average_precision_score(y[te], prob),
            })
        print(f"  [fold {fold} done] {ds_name}", flush=True)

    df = pd.DataFrame(rows)
    summ = df.groupby("model").agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
    ).reset_index()
    summ.insert(0, "dataset", ds_name)
    summ["test_minor_ratio"] = min_ratio
    summ["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[{ds_name} done] n={n} min_ratio={min_ratio:.4f} "
          f"elapsed={summ['elapsed_s'].iloc[0]}s", flush=True)
    return summ, df


def main():
    t_all = time.time()
    # probe xgboost availability
    try:
        import xgboost
        AVAIL["XGBoost"] = xgboost.__version__
    except Exception:
        AVAIL["XGBoost"] = None
    print("Library availability:", {"xgboost": AVAIL.get("XGBoost") or "MISSING",
                                    "EBM": "handcrafted (no dependency)"}, flush=True)

    parts = []
    fold_parts = []
    for ds in DATASETS:
        summ, folds = run_dataset(ds)
        parts.append(summ)
        fold_parts.append(folds)
    out = pd.concat(parts, ignore_index=True)
    fold_df = pd.concat(fold_parts, ignore_index=True)

    # merge the own models (PCBC / PCBC-MS-H3): the CSV holds one aggregated *_mean row per dataset, select and concat
    COMMON = ["dataset", "model",
              "precision_mean", "precision_std",
              "recall_mean", "recall_std",
              "f1_mean", "f1_std",
              "auc_mean", "auc_std",
              "auc_pr_mean", "auc_pr_std",
              "test_minor_ratio", "elapsed_s"]
    out = out[COMMON]
    if os.path.exists(OWN_CSV):
        own = pd.read_csv(OWN_CSV)
        own = own[own["model"].isin(["PCBC", "PCBC-MS-H3"])].copy()
        own = own[COMMON]
        out = pd.concat([out, own], ignore_index=True)
    else:
        print("WARNING: OWN_CSV not found, PCBC/PCBC-MS-H3 rows will be absent", flush=True)

    out.to_csv(RESULT_CSV, index=False)

    # nine-dataset means (model x metric)
    means = out.groupby("model")[["precision_mean", "recall_mean", "f1_mean",
                                  "auc_mean", "auc_pr_mean"]].mean()
    order = ["LR", "XGBoost", "RF", "EBM", "PCBC", "PCBC-MS-H3"]
    order = [m for m in order if m in means.index]
    print("\n===== 9-dataset means by model =====")
    print(means.reindex(order).round(4).to_string())
    print(f"\n[Saved] {RESULT_CSV}  |  total {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
