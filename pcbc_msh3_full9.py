# -*- coding: utf-8 -*-
"""
PCBC main experiment (paper Table III, nine static datasets)
============================================================
Self-contained reproduction of the main results. No external project modules
are required: the data loader, the Delaunay residual-correction learner and the
PCBC-1 (single-layer) / PCBC (hierarchical serial, K=3) models are all defined
below. The only runtime dependencies are numpy / pandas / scipy /
scikit-learn; datasets are read from ../datasets/ relative to this file.

Models (naming matches the paper, Table III):
  - PCBC-1 : single-layer variant, K=1 parallel single subspace (ablation baseline)
  - PCBC   : full serial chain, K=3 hierarchical re-centering (proposed model)

Protocol (identical to the paper):
  - nine datasets (Table I); label = last CSV column; minority class positive
  - paired stratified 5-fold CV, StratifiedKFold(5, shuffle=True, random_state=42)
  - train-only z-score normalisation; decision threshold 0.5; alpha = 0.6
  - candidate-pair budget by sample size: n>=1e5 -> 12, n>=3e4 -> 20, else 30

Output: pcbc_msh3_full9_results.csv (written to the repository root)

Run:  python pcbc_msh3_full9.py        (from inside the code/ folder)
"""
import os
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from scipy.spatial import Delaunay

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_msh3_full9_results.csv")

ALPHA = 0.6          # fusion weight of the geometric correction (paper default)
TOP_K = 30           # default candidate-pair budget (overridden per dataset)

# Paper Table III datasets (same order as the table)
DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]


# ------------------------- utilities -------------------------
def logit_to_prob(l):
    return 1.0 / (1.0 + np.exp(-np.clip(l, -50, 50)))


def load_dataset(ds_name):
    """Load a dataset; return X, y, minority_label, min_ratio."""
    df = pd.read_csv(os.path.join(DATA_DIR, "datasets", f"{ds_name}.csv"))
    label_col = df.columns[-1]
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col]).astype(np.float32).values
    uniq, cnt = np.unique(y, return_counts=True)
    minority_label = uniq[np.argmin(cnt)]
    min_ratio = cnt.min() / len(y)
    return X, y, minority_label, min_ratio


def barycentric_coords(tri, simplex_idx, points):
    """
    Barycentric coordinates of points within their host simplices
    (distance-decay weights, Eq. (4) of the paper). points: (n,2).
    Returns (n,3) coordinates summing to one.
    """
    T = tri.transform[simplex_idx]              # (n, 2, 3): inverse basis + shift
    delta = points - T[:, 2]                    # (n, 2)
    lam2 = np.einsum("nij,nj->ni", T[:, :2], delta)  # (n, 2)
    lam = np.hstack([lam2, 1.0 - lam2.sum(axis=1, keepdims=True)])
    lam = np.clip(lam, 0.0, 1.0)
    s = lam.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    return lam / s


# ------------------------- Delaunay residual-correction learner -------------------------
class TemporalLSIM:
    """
    Delaunay-based local residual correction on a selected 2-D feature pair.
    use_bary=True: true barycentric (distance-decay) interpolation.
    """

    def __init__(self, pair, use_bary=True, seed=42):
        self.pair = pair
        self.use_bary = use_bary
        self.seed = seed
        self.tri = None
        self.xy_train = None
        self.resid_train = None
        self.fill = 0.0

    def fit(self, X, resid, y, year_id):
        xy = X[:, self.pair].copy()
        rng = np.random.default_rng(self.seed)
        xy += rng.standard_normal(xy.shape) * 1e-6   # avoid collinear degeneracy
        self.fill = float(np.mean(resid))            # out-of-hull fallback
        self.tri = Delaunay(xy)
        self.xy_train = xy
        self.resid_train = resid
        return self

    def predict(self, X):
        xy_query = X[:, self.pair]
        tri_idx = self.tri.find_simplex(xy_query)
        pred = np.full(len(xy_query), self.fill)
        valid = np.nonzero(tri_idx != -1)[0]
        if len(valid) == 0:
            return pred
        if self.use_bary:
            lam_all = barycentric_coords(self.tri, tri_idx[valid], xy_query[valid])
        for k, i in enumerate(valid):
            verts = self.tri.simplices[tri_idx[i]]
            r = self.resid_train[verts]
            if self.use_bary:
                w = lam_all[k]
            else:
                w = np.ones(3)
            sw = w.sum()
            pred[i] = np.sum(r * w) / sw if sw > 1e-12 else self.fill
        return pred


# ------------------------- candidate-pair selection (nested CV, debiased) -------------------------
def select_top_k_pairs(X_tr, y_tr, lr_coef, top_k=TOP_K, K=3, seed=42, target_resid=None):
    """
    Rank feature pairs by |coef_i * coef_j|, then nested-3-fold CV on the
    residual-interpolation gain; return the top-K pairs (and full ranking).
    target_resid: for hierarchical mode, the residual to correct at this layer.
    """
    D = X_tr.shape[1]
    coef = np.abs(lr_coef)
    cand = {}
    for i in range(D):
        for j in range(i + 1, D):
            cand[(i, j)] = coef[i] * coef[j]
    top = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:top_k]
    inner_cv = StratifiedKFold(3, shuffle=True, random_state=42)
    pair_scores = {}
    for (i, j), _ in top:
        aucs = []
        for itr, ival in inner_cv.split(X_tr, y_tr):
            lr = LogisticRegression(max_iter=500, class_weight="balanced")
            lr.fit(X_tr[itr], y_tr[itr])
            if target_resid is None:
                resid = y_tr[itr] - lr.predict_proba(X_tr[itr])[:, 1]
            else:
                resid = target_resid[itr]      # hierarchical: this layer's residual
            lsim = TemporalLSIM((i, j), use_bary=False, seed=seed)
            lsim.fit(X_tr[itr], resid, y_tr[itr], np.ones(len(y_tr[itr])))
            logit = lr.decision_function(X_tr[ival]) + 0.5 * lsim.predict(X_tr[ival])
            prob = logit_to_prob(logit)
            try:
                aucs.append(roc_auc_score(y_tr[ival], prob))
            except Exception:
                aucs.append(0.5)
        pair_scores[(i, j)] = np.mean(aucs)
    ranked = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:K]], ranked


# ------------------------- PCBC models -------------------------
class PCBC_MS(BaseEstimator, ClassifierMixin):
    """
    mode='parallel': average the corrections of the top-K subspaces.
    mode='hier':     serial chain; each layer re-fits the residual left by the
                    previous layers (PCBC, K=3 in the paper).
    """

    def __init__(self, alpha=ALPHA, seed=42, K=3, mode="parallel", use_bary=True):
        self.alpha = alpha
        self.seed = seed
        self.K = K
        self.mode = mode
        self.use_bary = use_bary
        self.lr = None
        self.pairs = None
        self.inter = None
        self.layers = None
        self.fill = 0.0

    def fit(self, X, y, year_id=None, pairs=None):
        if year_id is None:
            year_id = np.ones(len(y))
        self.lr = LogisticRegression(max_iter=500, class_weight="balanced",
                                     random_state=self.seed)
        self.lr.fit(X, y)
        if self.mode == "parallel":
            if pairs is None:
                pairs, _ = select_top_k_pairs(X, y, self.lr.coef_[0],
                                              top_k=TOP_K, K=self.K, seed=self.seed)
            self.pairs = pairs
            resid = y - self.lr.predict_proba(X)[:, 1]
            self.inter = _MultiLSIM(pairs, use_bary=self.use_bary, seed=self.seed)
            self.inter.fit(X, resid, y, year_id)
        else:  # hier: serial residual re-centering
            prob = self.lr.predict_proba(X)[:, 1]
            self.layers = []
            for k in range(self.K):
                resid_k = y - prob
                pair_k, _ = select_top_k_pairs(X, y, self.lr.coef_[0],
                                               top_k=TOP_K, K=1, seed=self.seed,
                                               target_resid=resid_k)
                sim = TemporalLSIM(pair_k[0], use_bary=self.use_bary, seed=self.seed)
                sim.fit(X, resid_k, y, year_id)
                self.layers.append(sim)
                logit = self.lr.decision_function(X) + self.alpha * sim.predict(X)
                prob = logit_to_prob(logit)
        return self

    def decision_function(self, X):
        logit = self.lr.decision_function(X)
        if self.mode == "parallel":
            logit += self.alpha * self.inter.predict(X)
        else:
            for sim in self.layers:
                logit += self.alpha * sim.predict(X)
        return logit

    def predict_proba(self, X):
        p = logit_to_prob(self.decision_function(X))
        return np.vstack([1 - p, p]).T


class _MultiLSIM:
    """Averaged correction of K independent 2-D learners (parallel mode)."""

    def __init__(self, pairs, use_bary=True, seed=42):
        self.sims = [TemporalLSIM(pair=p, use_bary=use_bary, seed=seed)
                     for p in pairs]

    def fit(self, X, resid, y, year_id=None):
        if year_id is None:
            year_id = np.ones(len(y))
        for s in self.sims:
            s.fit(X, resid, y, year_id)
        return self

    def predict(self, X):
        return np.mean([s.predict(X) for s in self.sims], axis=0)


# ------------------------- main experiment -------------------------
def effective_topk(n):
    if n >= 100000:
        return 12
    if n >= 30000:
        return 20
    return 30


def run_dataset(ds_name):
    global TOP_K
    t0 = time.time()
    X, y, minority_label, min_ratio = load_dataset(ds_name)
    n = len(y)
    topk = effective_topk(n)
    TOP_K = topk            # candidate-pair budget for this dataset
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    specs = {"PCBC-1": dict(K=1, mode="parallel"),
             "PCBC": dict(K=3, mode="hier")}
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        for vname, kw in specs.items():
            m = PCBC_MS(seed=42, **kw)
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
    summ.insert(1, "n_samples", n)
    summ["test_minor_ratio"] = min_ratio
    summ["top_k"] = topk
    summ["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[{ds_name} done] n={n} min_ratio={min_ratio:.4f} topk={topk} "
          f"elapsed={summ['elapsed_s'].iloc[0]}s", flush=True)
    return summ


def main():
    t_all = time.time()
    parts = []
    for ds in DATASETS:
        parts.append(run_dataset(ds))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RESULT_CSV, index=False)
    print(out.round(4).to_string(index=False), flush=True)
    print(f"\n[Saved] {RESULT_CSV}  |  total {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
