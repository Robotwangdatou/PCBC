# -*- coding: utf-8 -*-
"""
Direction 4 Experiment: Debiased Hierarchical Multi-Subspace PCBC
================================================================
Motivation: the direction-1 conclusion - PCBC cannot catch XGBoost because of insufficient capacity (one 2D subspace plus a linear baseline).
Direction 4 = upgrade the single-subspace geometric correction to a multi-subspace ensemble, adding capacity while reducing the validation bias of single-pair selection.

Architecture (same data, same global LR, controlled variables):
  PCBC        : original reproduction (one subspace, single best pair by nested CV)
  PCBC-MS3    : parallel top-3 subspace ensemble (the top three feature pairs by AUC ranking, each with its own Delaunay + residual interpolation, delta = mean)
  PCBC-MS5    : parallel top-5 subspace ensemble
  PCBC-MS-H3  : hierarchical serial correction (layer k re-selects the pair for that layer's residual, layerwise re-centering, delta = sum_k alpha*delta_k)

Debiasing mechanisms (material for the exposition in the paper):
  (1) top-K ensemble: nested-CV selection of a single pair overfits the validation split (selection bias);
    averaging several subspaces flattens the variance of which subspace is chosen (bagging-style variance reduction).
  (2) hierarchical serial: each layer corrects the residual left by the previous layer (residual re-centering),
    approaching the irreducible noise and cancelling the systematic bias of a single correction.

Evaluation:
  - temporal task 4 (train Y1-4 -> test Y5), 5 seeds, same protocol as direction 1, XGB/EBM cited from the reference table
  - static polish_Bankruptcy5, 5-fold (comparable to the static nine-dataset protocol of Tables 2 and 3 of the paper)

Run: python pcbc_multisubspace_experiment.py
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
)
import importlib.util

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_multisubspace_results.csv")
REFERENCE_CSV = os.path.join(DATA_DIR, "temporal_extrapolation_all_models.csv")

ALPHA = 0.6
TOP_K = 30
N_REPEAT = 5

# ------------------------- top-K subspace selection (debiasing: the top-K pairs by AUC ranking) -------------------------
def select_top_k_pairs(X_tr, y_tr, lr_coef, top_k=TOP_K, K=3, seed=42, target_resid=None):
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
                resid = target_resid[itr]   # hierarchical mode: the correction target is that layer's residual
            lsim = exp1.TemporalLSIM((i, j), use_bary=False, seed=seed)
            lsim.fit(X_tr[itr], resid, y_tr[itr], np.ones(len(y_tr[itr])))
            logit = lr.decision_function(X_tr[ival]) + 0.5 * lsim.predict(X_tr[ival])
            prob = exp1.logit_to_prob(logit)
            try:
                aucs.append(roc_auc_score(y_tr[ival], prob))
            except Exception:
                aucs.append(0.5)
        pair_scores[(i, j)] = np.mean(aucs)
    ranked = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:K]], ranked

# ------------------------- parallel multi-subspace ensemble module -------------------------
class MultiLSIM:
    """Each of the K subspaces builds its own Delaunay + residual interpolation; the averaged correction is returned (bagging-style)."""
    def __init__(self, pairs, use_bary=True, seed=42):
        self.sims = [exp1.TemporalLSIM(pair=p, use_bary=use_bary, seed=seed)
                     for p in pairs]

    def fit(self, X, resid, y, year_id=None):
        if year_id is None:
            year_id = np.ones(len(y))
        for s in self.sims:
            s.fit(X, resid, y, year_id)
        return self

    def predict(self, X):
        return np.mean([s.predict(X) for s in self.sims], axis=0)

# ------------------------- the PCBC multi-subspace model -------------------------
class PCBC_MS(BaseEstimator, ClassifierMixin):
    """
    mode='parallel': the top-K pairs are ensembled in parallel
    mode='hier':     serial layer by layer (each layer re-selects the single best pair for that layer's residual)
    """
    def __init__(self, alpha=ALPHA, seed=42, K=3, mode="parallel", use_bary=True):
        self.alpha = alpha
        self.seed = seed
        self.K = K
        self.mode = mode
        self.use_bary = use_bary
        self.lr = None
        self.pairs = None
        self.layers = None   # only in hierarchical mode
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
            self.inter = MultiLSIM(pairs, use_bary=self.use_bary, seed=self.seed)
            self.inter.fit(X, resid, y, year_id)
        else:  # hier
            prob = self.lr.predict_proba(X)[:, 1]
            self.layers = []
            for k in range(self.K):
                resid_k = y - prob
                pair_k, _ = select_top_k_pairs(X, y, self.lr.coef_[0],
                                               top_k=TOP_K, K=1, seed=self.seed,
                                               target_resid=resid_k)
                sim = exp1.TemporalLSIM(pair_k[0], use_bary=self.use_bary,
                                        seed=self.seed)
                sim.fit(X, resid_k, y, year_id)
                self.layers.append(sim)
                # update the current cumulative probability (layerwise re-centering)
                logit = self.lr.decision_function(X) + self.alpha * sim.predict(X)
                prob = exp1.logit_to_prob(logit)
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
        p = exp1.logit_to_prob(self.decision_function(X))
        return np.vstack([1 - p, p]).T

# ------------------------- temporal task 4 (same protocol as direction 1, 5 seeds) -------------------------
def eval_temporal_task4():
    train_ds = ["polish_Bankruptcy1", "polish_Bankruptcy2",
                "polish_Bankruptcy3", "polish_Bankruptcy4"]
    Xs, ys = [], []
    for ds in train_ds:
        X, y, _, _ = exp1.load_dataset(ds)
        Xs.append(X); ys.append(y)
    X_train = np.vstack(Xs)
    y_train = np.hstack(ys)
    X_test, y_test, minority_label, test_min_ratio = exp1.load_dataset("polish_Bankruptcy5")
    mu = X_train.mean(0); sigma = X_train.std(0) + 1e-6
    X_tr_n = (X_train - mu) / sigma
    X_te_n = (X_test - mu) / sigma

    # cache: screen once per training set (top-5 yields all of them, MS3 takes the first three)
    lr0 = LogisticRegression(max_iter=500, class_weight="balanced").fit(X_tr_n, y_train)
    pairs5, _ = select_top_k_pairs(X_tr_n, y_train, lr0.coef_[0],
                                   top_k=TOP_K, K=5, seed=42)

    specs = {
        "PCBC":      dict(K=1, mode="parallel"),
        "PCBC-MS3":  dict(K=3, mode="parallel"),
        "PCBC-MS5":  dict(K=5, mode="parallel"),
        "PCBC-MS-H3":dict(K=3, mode="hier"),
    }
    records = []
    for seed in range(N_REPEAT):
        for vname, kw in specs.items():
            if kw["mode"] == "hier":
                pairs = None
            else:
                pairs = pairs5[:kw["K"]]
            m = PCBC_MS(seed=seed, **kw)
            m.fit(X_tr_n, y_train, pairs=pairs)
            prob = m.predict_proba(X_te_n)[:, 1]
            records.append((vname, seed, prob))

    rows = []
    for vname, seed, prob in records:
        y_pred = (prob >= 0.5).astype(int)
        prf = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
        uniq = np.unique(y_test)
        idx_min = 0 if uniq[0] == minority_label else 1
        rows.append({
            "task": 4, "model": vname, "seed": seed,
            "precision": prf[0][idx_min], "recall": prf[1][idx_min], "f1": prf[2][idx_min],
            "auc": roc_auc_score(y_test, prob), "auc_pr": average_precision_score(y_test, prob),
        })
    df = pd.DataFrame(rows)
    summ = df.groupby("model").agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
    ).reset_index()
    summ.insert(0, "task", 4)
    summ["test_set"] = "polish_Bankruptcy5"
    summ["test_minor_ratio"] = test_min_ratio
    return summ

# ------------------------- static Y5 (5-fold, same convention as the static nine-dataset protocol) -------------------------
def eval_static_y5():
    X, y, minority_label, min_ratio = exp1.load_dataset("polish_Bankruptcy5")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    specs = {"PCBC": dict(K=1, mode="parallel"),
             "PCBC-MS3": dict(K=3, mode="parallel"),
             "PCBC-MS5": dict(K=5, mode="parallel")}
    records = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        lr0 = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xtr, y[tr])
        pairs5, _ = select_top_k_pairs(Xtr, y[tr], lr0.coef_[0],
                                       top_k=TOP_K, K=5, seed=42)
        for vname, kw in specs.items():
            pairs = pairs5[:kw["K"]] if kw["K"] > 1 else None
            m = PCBC_MS(seed=42, **kw)
            m.fit(Xtr, y[tr], pairs=pairs)
            prob = m.predict_proba(Xte)[:, 1]
            y_pred = (prob >= 0.5).astype(int)
            prf = precision_recall_fscore_support(y[te], y_pred, average=None, zero_division=0)
            uniq = np.unique(y[te])
            idx_min = 0 if uniq[0] == minority_label else 1
            records.append({
                "fold": fold, "model": vname,
                "precision": prf[0][idx_min], "recall": prf[1][idx_min], "f1": prf[2][idx_min],
                "auc": roc_auc_score(y[te], prob), "auc_pr": average_precision_score(y[te], prob),
            })
    df = pd.DataFrame(records)
    summ = df.groupby("model").agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
    ).reset_index()
    summ.insert(0, "task", "Y5-static")
    summ["test_set"] = "polish_Bankruptcy5"
    summ["test_minor_ratio"] = min_ratio
    return summ

# ------------------------- main -------------------------
def main():
    t0 = time.time()
    print("===== Temporal Task4 (Y1-4 -> Y5) =====", flush=True)
    t4 = eval_temporal_task4()
    print(t4.round(4).to_string(index=False), flush=True)
    print("===== Static Y5 (5-fold) =====", flush=True)
    st = eval_static_y5()
    print(st.round(4).to_string(index=False), flush=True)

    out = pd.concat([t4, st], ignore_index=True)

    # merge the temporal reference table of the original paper (XGB / EBM)
    if os.path.exists(REFERENCE_CSV):
        ref = pd.read_csv(REFERENCE_CSV)
        ref = ref[ref["model"].isin(["XGB", "EBM"])].copy()
        ref = ref.rename(columns={"task_id": "task"})
        ref["model"] = ref["model"].replace({"XGB": "XGBoost(ref)", "EBM": "EBM(ref)"})
        ref = ref[ref["task"] == 4].copy()
        out = pd.concat([out, ref[["task", "model", "precision_mean", "precision_std",
                                   "f1_mean", "f1_std", "auc_mean", "auc_std",
                                   "auc_pr_mean", "auc_pr_std", "test_set",
                                   "test_minor_ratio"]]], ignore_index=True)
    out.to_csv(RESULT_CSV, index=False)
    print(f"\n[Saved] {RESULT_CSV}  |  total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
