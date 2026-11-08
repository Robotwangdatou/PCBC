# -*- coding: utf-8 -*-
"""
Direction 1 experiment: dynamic temporal Delaunay (time-adaptive geometry) for PCBC
=======================================================================
Paper: Hard-Boundary Geometric Calibration for Auditable Credit Scoring (rejected)
Goal : check whether time-adaptive geometry can repair the temporal weakness of Table 3 (temporal tasks dominated by XGBoost)

Model variants (same rolling protocol, controlled variables):
  PCBC        : original reproduction (matching Table 10 of the paper, temporal_extrapolation_lsim.csv)
  PCBC-Geo    : true barycentric coordinates lambda_geo (distance decay) replacing the vertex-residual averaging of the original
  PCBC-TW     : barycentric coordinates + temporal decay weights w_t = exp(-gamma*(T-t))   <- core of direction 1
  PCBC-TW-Win2: barycentric coordinates + temporal decay + training on the last two years only (aggressive grid adaptation)

Rolling protocol (identical to the original paper):
  Task1: train Y1        -> test Y2
  Task2: train Y1+Y2     -> test Y3
  Task3: train Y1+Y2+Y3  -> test Y4
  Task4: train Y1..Y4    -> test Y5
  N_REPEAT random-seed repetitions, reported as mean +/- std

Metrics: Precision / F1 (aligned with the original paper) + AUC / PR-AUC (the ranking metrics missing from the original draft)

Run: python pcbc_temporal_direction1.py
"""
import os
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
)
from scipy.spatial import Delaunay

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_temporal_direction1_results.csv")
# temporal reference table of the original paper (XGB/EBM under the same protocol, cited directly to avoid environment dependencies)
REFERENCE_CSV = os.path.join(DATA_DIR, "temporal_extrapolation_all_models.csv")

ALPHA = 0.6          # fusion weight (identical to the original paper)
TOP_K = 30           # number of candidate pairs retained by the sparse screen
N_REPEAT = 5         # number of random-seed repetitions (the original paper used 10; 5 here for run time, adjustable)
GAMMA = 0.3          # temporal decay rate (main reported value; sensitivity below)
TIME_GAMMAS = [0.1, 0.3, 0.5]  # sensitivity sweep of the temporal decay

# ------------------------- utilities -------------------------
def logit_to_prob(l):
    return 1.0 / (1.0 + np.exp(-np.clip(l, -50, 50)))

def load_dataset(ds_name):
    """Load a single-year dataset; return X, y, minority_label, min_ratio."""
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
    Computes the barycentric coordinates of points inside their host simplex (distance-decay weights, lambda of Eq. 4 in the paper).
    points: (n,2), each point known to lie in the triangle simplex_idx
    Returns: (n,3) barycentric coordinates summing to one
    """
    T = tri.transform[simplex_idx]              # (n, 2, 3): first two columns = inverse basis, third = shift
    delta = points - T[:, 2]                    # (n, 2)
    lam2 = np.einsum("nij,nj->ni", T[:, :2], delta)  # (n, 2)
    lam = np.hstack([lam2, 1.0 - lam2.sum(axis=1, keepdims=True)])
    lam = np.clip(lam, 0.0, 1.0)                # numerical guard
    s = lam.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    return lam / s

# ------------------------- core interactive module -------------------------
class TemporalLSIM:
    """
    Time-adaptive Delaunay local residual-correction module.
    Optional mechanisms:
      use_bary     : true barycentric coordinates (distance decay) instead of vertex averaging
      use_time     : temporal decay weights w_t = exp(-gamma*(T - t)) modulating the aggregation
      win_years    : if >0, train only on the last win_years years (grid adaptation)
    """
    def __init__(self, pair, use_bary=True, use_time=False, gamma=0.3,
                 win_years=0, seed=42):
        self.pair = pair
        self.use_bary = use_bary
        self.use_time = use_time
        self.gamma = gamma
        self.win_years = win_years
        self.seed = seed
        self.tri = None
        self.xy_train = None
        self.resid_train = None
        self.time_weight = None     # temporal decay weight of each training record
        self.fill = 0.0

    def fit(self, X, resid, y, year_id):
        """
        X: training features (after standardisation); resid: residual; y: labels;
        year_id: year index of each record (1..T), T = latest year
        """
        mask = np.ones(len(y), dtype=bool)
        if self.win_years and self.win_years > 0:
            T_max = year_id.max()
            mask = (year_id >= T_max - self.win_years + 1) & mask
        Xw, residw, yw, yearw = X[mask], resid[mask], y[mask], year_id[mask]

        xy = Xw[:, self.pair].copy()
        rng = np.random.default_rng(self.seed)
        xy += rng.standard_normal(xy.shape) * 1e-6   # avoids degenerate collinear simplices (same as the original)
        self.fill = np.mean(residw)                   # fallback value outside the hull (same as the original)

        self.tri = Delaunay(xy)
        self.xy_train = xy
        self.resid_train = residw

        # temporal decay weights (meaningful only when the training years span more than one)
        if self.use_time:
            T_max = yearw.max()
            self.time_weight = np.exp(-self.gamma * (T_max - yearw))
        else:
            self.time_weight = np.ones(len(yearw))

    def predict(self, X):
        xy_query = X[:, self.pair]
        tri_idx = self.tri.find_simplex(xy_query)
        pred = np.full(len(xy_query), self.fill)
        valid = np.nonzero(tri_idx != -1)[0]
        if len(valid) == 0:
            return pred

        if self.use_bary:
            lam_all = barycentric_coords(self.tri, tri_idx[valid], xy_query[valid])
        # original behaviour: w_vals = class weights (=1), equivalent to plain vertex averaging
        for k, i in enumerate(valid):
            verts = self.tri.simplices[tri_idx[i]]
            r = self.resid_train[verts]
            if self.use_bary:
                w = lam_all[k]
            else:
                w = np.ones(3)
            if self.use_time:
                w = w * self.time_weight[verts]
            sw = w.sum()
            pred[i] = np.sum(r * w) / sw if sw > 1e-12 else self.fill
        return pred

# ------------------------- sparse interaction screening (same as the original, returns the best pair) -------------------------
def select_best_pair_nested(X_tr, y_tr, lr_coef, top_k=TOP_K, seed=42):
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
            resid = y_tr[itr] - lr.predict_proba(X_tr[itr])[:, 1]
            lsim = TemporalLSIM((i, j), use_bary=False, seed=seed)
            lsim.fit(X_tr[itr], resid, y_tr[itr], np.ones(len(y_tr[itr])))
            logit = lr.decision_function(X_tr[ival]) + 0.5 * lsim.predict(X_tr[ival])
            prob = logit_to_prob(logit)
            try:
                aucs.append(roc_auc_score(y_tr[ival], prob))
            except Exception:
                aucs.append(0.5)
        pair_scores[(i, j)] = np.mean(aucs)
    return max(pair_scores, key=pair_scores.get)

# ------------------------- PCBC model (configurable geometry) -------------------------
class PCBC(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=ALPHA, seed=42, use_bary=False, use_time=False,
                 gamma=GAMMA, win_years=0, lr_time_weight=False):
        self.alpha = alpha
        self.seed = seed
        self.use_bary = use_bary
        self.use_time = use_time
        self.gamma = gamma
        self.win_years = win_years
        self.lr_time_weight = lr_time_weight   # True: the time weights enter the LR training layer (sample_weight)
        self.lr = None
        self.pair = None
        self.inter = None
        self.year_id = None

    def fit(self, X, y, year_id=None, pair=None):
        if year_id is None:
            year_id = np.ones(len(y))
        if self.lr_time_weight and self.use_time:
            T_max = year_id.max()
            sw = np.exp(-self.gamma * (T_max - year_id))
        else:
            sw = None
        self.lr = LogisticRegression(max_iter=500, class_weight="balanced",
                                     random_state=self.seed)
        self.lr.fit(X, y, sample_weight=sw)
        if pair is None:
            pair = select_best_pair_nested(X, y, self.lr.coef_[0],
                                           top_k=TOP_K, seed=self.seed)
        self.pair = pair
        prob = self.lr.predict_proba(X)[:, 1]
        resid = y - prob
        self.inter = TemporalLSIM(
            pair=self.pair, use_bary=self.use_bary, use_time=self.use_time,
            gamma=self.gamma, win_years=self.win_years, seed=self.seed,
        )
        self.inter.fit(X, resid, y, year_id)
        self.year_id = year_id
        return self

    def decision_function(self, X):
        logit = self.lr.decision_function(X)
        logit += self.alpha * self.inter.predict(X)
        return logit

    def predict_proba(self, X):
        logit = self.decision_function(X)
        p = logit_to_prob(logit)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ------------------------- baseline models -------------------------
def make_lr(seed):
    return LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)

def make_dt(seed):
    return DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)

# XGB / EBM are not run here: the numbers of temporal_extrapolation_all_models.csv are cited directly
# (same protocol and configuration, XGB: n_estimators=50, max_depth=3, scale_pos_weight; EBM: pure-Python implementation)

# ------------------------- single-task evaluation -------------------------
def eval_task(train_ds_list, test_ds_name, task_id, n_repeat=N_REPEAT):
    X_tr_all, y_tr_all, year_id_all = [], [], []
    for idx, ds in enumerate(train_ds_list, start=1):
        X_ds, y_ds, _, _ = load_dataset(ds)
        X_tr_all.append(X_ds)
        y_tr_all.append(y_ds)
        year_id_all.append(np.full(len(y_ds), idx))
    X_train = np.vstack(X_tr_all)
    y_train = np.hstack(y_tr_all)
    year_id = np.hstack(year_id_all)
    X_test, y_test, minority_label, test_min_ratio = load_dataset(test_ds_name)

    # standardisation with training-set statistics (prevents leakage)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-6
    X_tr_norm = (X_train - mu) / sigma
    X_te_norm = (X_test - mu) / sigma

    # pair-screening cache: screen once per training set (the jitter seed is fixed at 42 for determinism), reused by the five seeds
    # time adaptation (win_years) changes the training set, so it needs its own screening
    pair_cache = {}
    def get_pair(X_set, y_set, win_years):
        key = (X_set.shape[0], win_years)
        if key not in pair_cache:
            pair_cache[key] = select_best_pair_nested(X_set, y_set,
                                                      LogisticRegression(
                                                          max_iter=500, class_weight="balanced"
                                                      ).fit(X_set, y_set).coef_[0],
                                                      top_k=TOP_K, seed=42)
        return pair_cache[key]

    model_specs = {
        "PCBC":         dict(use_bary=False, use_time=False),
        "PCBC-Geo":     dict(use_bary=True,  use_time=False),
        "PCBC-TW":      dict(use_bary=True,  use_time=True,  gamma=GAMMA),
        "PCBC-TW-LR":   dict(use_bary=True,  use_time=True,  gamma=GAMMA, lr_time_weight=True),
        "PCBC-TW-Win2": dict(use_bary=True,  use_time=True,  gamma=GAMMA, win_years=2),
    }
    # pre-screen pairs on the full training set
    pair_full = get_pair(X_tr_norm, y_train, 0)
    # pre-screen pairs on the windowed training set (win_years=2)
    mask_w = year_id >= year_id.max() - 1
    pair_win2 = get_pair(X_tr_norm[mask_w], y_train[mask_w], 2)

    records = []
    for seed in range(n_repeat):
        # --- baselines (LR / DT implemented here; XGB / EBM cited from the reference table) ---
        for name, fn in [("LR", make_lr), ("DT", make_dt)]:
            model = fn(seed)
            model.fit(X_tr_norm, y_train)
            prob = model.predict_proba(X_te_norm)[:, 1]
            records.append((task_id, name, seed, prob, y_test))
        # --- PCBC variants ---
        for vname, kw in model_specs.items():
            p = pair_win2 if kw.get("win_years", 0) > 0 else pair_full
            model = PCBC(seed=seed, **kw)
            model.fit(X_tr_norm, y_train, year_id=year_id, pair=p)
            prob = model.predict_proba(X_te_norm)[:, 1]
            records.append((task_id, vname, seed, prob, y_test))

    # aggregate metrics
    rows = []
    for task_id, mname, seed, prob, yt in records:
        y_pred = (prob >= 0.5).astype(int)
        prf = precision_recall_fscore_support(yt, y_pred, average=None, zero_division=0)
        uniq = np.unique(yt)
        idx_min = 0 if uniq[0] == minority_label else 1
        rows.append({
            "task": task_id, "model": mname, "seed": seed,
            "precision": prf[0][idx_min], "recall": prf[1][idx_min], "f1": prf[2][idx_min],
            "auc": roc_auc_score(yt, prob),
            "auc_pr": average_precision_score(yt, prob),
        })
    df = pd.DataFrame(rows)
    summary = df.groupby(["task", "model"]).agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
    ).reset_index()
    summary["test_set"] = test_ds_name
    summary["test_minor_ratio"] = test_min_ratio
    return summary, df

# ------------------------- main -------------------------
def main():
    tasks = [
        (["polish_Bankruptcy1"], "polish_Bankruptcy2", 1),
        (["polish_Bankruptcy1", "polish_Bankruptcy2"], "polish_Bankruptcy3", 2),
        (["polish_Bankruptcy1", "polish_Bankruptcy2", "polish_Bankruptcy3"],
         "polish_Bankruptcy4", 3),
        (["polish_Bankruptcy1", "polish_Bankruptcy2", "polish_Bankruptcy3",
          "polish_Bankruptcy4"], "polish_Bankruptcy5", 4),
    ]
    all_summary = []
    t0 = time.time()
    for train_list, test_name, tid in tasks:
        print(f"\n===== Task{tid}: train {len(train_list)}y -> test {test_name} =====", flush=True)
        summ, _ = eval_task(train_list, test_name, tid)
        all_summary.append(summ)
        print(summ.round(4).to_string(index=False), flush=True)
    out = pd.concat(all_summary, ignore_index=True)

    # merge the reference table of the original paper (XGB / EBM, same protocol)
    if os.path.exists(REFERENCE_CSV):
        ref = pd.read_csv(REFERENCE_CSV)
        ref = ref[ref["model"].isin(["XGB", "EBM"])].copy()
        ref = ref.rename(columns={"task_id": "task"})
        ref["model"] = ref["model"].replace({"XGB": "XGBoost(ref)", "EBM": "EBM(ref)"})
        ref_rows = []
        for _, r in ref.iterrows():
            ref_rows.append({
                "task": r["task"], "model": r["model"],
                "precision_mean": r["precision_mean"], "precision_std": r["precision_std"],
                "f1_mean": r["f1_mean"], "f1_std": r["f1_std"],
                "auc_mean": r["auc_mean"], "auc_std": r["auc_std"],
                "auc_pr_mean": r["auc_pr_mean"], "auc_pr_std": r["auc_pr_std"],
                "test_set": r["test_set"], "test_minor_ratio": r["test_minor_ratio"],
            })
        out = pd.concat([out, pd.DataFrame(ref_rows)], ignore_index=True)
    out.to_csv(RESULT_CSV, index=False)
    print(f"\n[Saved] {RESULT_CSV}  |  total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
