# -*- coding: utf-8 -*-
"""
Hard-Boundary Ablation: Gated (fallback) vs. NoGate (extrapolation)
===================================================================
P1 experiment: hard-boundary ablation (gated versus ungated)

Core mechanism:
  - Gated   (= the existing PCBC): a query outside the convex hull falls back to the global baseline fill (hard boundary, no extrapolation)
  - NoGate  (ablation variant)  : a query outside the convex hull is assigned to the simplex of its nearest training vertex
                            and interpolated with unclipped barycentric coordinates (negative weights allowed), i.e. extrapolation

Test bed: temporal rolling protocol on the Polish data (tasks 1..4, train Y1..Yk -> test Y{k+1})
Expectation: distribution drift pushes many test points outside the hull; NoGate extrapolation should produce extreme corrections and collapse the performance,
      while Gated falls back to the baseline and stays stable. hull_out_rate reports the out-of-hull trigger rate per task.

Run:
  python pcbc_nogate.py
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score,
                             precision_recall_fscore_support)
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import importlib.util

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BASE)
_spec = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_BASE, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)

DATA_DIR = _REPO
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_nogate_smoke.csv")

ALPHA = getattr(exp1, "ALPHA", 0.6)
TOP_K = 30
N_REPEAT = 5          # matches the paired five-fold protocol of the main experiment (final version)


# ------------------------- LSIM with a gate switch -------------------------
class LSIMGate:
    """
    Same Delaunay + barycentric interpolation as exp1.TemporalLSIM;
    gate=True  : outside the hull, fall back to fill (the existing PCBC behaviour)
    gate=False : outside the hull, run unclipped barycentric interpolation on the nearest simplex (extrapolation, negative weights allowed)
    """
    def __init__(self, pair, gate=True, use_bary=True, seed=42):
        self.pair = pair
        self.gate = gate
        self.use_bary = use_bary
        self.seed = seed
        self.tri = None
        self.xy_train = None
        self.resid_train = None
        self.fill = 0.0

    def fit(self, X, resid, y, year_id=None):
        if year_id is None:
            year_id = np.ones(len(y))
        self.fill = float(np.mean(resid))
        xy = X[:, self.pair].copy()
        rng = np.random.default_rng(self.seed)
        xy += rng.standard_normal(xy.shape) * 1e-6   # avoids degenerate collinear simplices
        self.tri = Delaunay(xy)
        self.xy_train = xy
        self.resid_train = resid
        return self

    def predict(self, X):
        xy_query = X[:, self.pair]
        tri_idx = self.tri.find_simplex(xy_query)
        pred = np.full(len(xy_query), self.fill)
        valid = np.nonzero(tri_idx != -1)[0]
        invalid = np.nonzero(tri_idx == -1)[0]
        # --- inside the hull: standard barycentric interpolation (same as the existing logic) ---
        if len(valid):
            if self.use_bary:
                lam_all = exp1.barycentric_coords(self.tri, tri_idx[valid], xy_query[valid])
            for k, i in enumerate(valid):
                verts = self.tri.simplices[tri_idx[i]]
                r = self.resid_train[verts]
                if self.use_bary:
                    w = lam_all[k]
                else:
                    w = np.ones(3)
                sw = w.sum()
                pred[i] = np.sum(r * w) / sw if sw > 1e-12 else self.fill
        # --- outside the hull: gated versus extrapolation ---
        if len(invalid):
            if self.gate:
                pred[invalid] = self.fill                 # hard-boundary fallback
            else:
                # nearest training vertex -> its associated simplex -> unclipped barycentric coordinates (extrapolation)
                dists = cdist(xy_query[invalid], self.xy_train)
                nearest_v = np.argmin(dists, axis=1)
                simps = self.tri.vertex_to_simplex[nearest_v]
                bad = simps == -1                          # defensive: a vertex with no associated simplex (very rare)
                keep = ~bad
                if bad.any():
                    pred[invalid[bad]] = self.fill
                if keep.any():
                    simps_k = simps[keep]
                    T = self.tri.transform[simps_k]
                    nan_row = np.isnan(T).any(axis=(1, 2))  # defensive: degenerate simplex
                    nan_idx = np.flatnonzero(nan_row)
                    ok_idx = np.flatnonzero(~nan_row)
                    if nan_idx.size:
                        pred[invalid[keep][nan_idx]] = self.fill
                    if ok_idx.size:
                        xy_k = xy_query[invalid[keep][ok_idx]]
                        T_ok = T[ok_idx]
                        simps_ok = simps_k[ok_idx]
                        delta = xy_k - T_ok[:, 2]
                        lam2 = np.einsum("nij,nj->ni", T_ok[:, :2], delta)
                        # the third coordinate is built as 1 - sum of the first two, so each row sums to one (negative weights allowed for extrapolation)
                        lam = np.hstack([lam2, 1.0 - lam2.sum(axis=1, keepdims=True)])
                        for k, i in enumerate(invalid[keep][ok_idx]):
                            verts = self.tri.simplices[simps_ok[k]]
                            r = self.resid_train[verts]
                            w = lam[k]
                            sw = w.sum()                  # equals one by construction of the barycentric coordinates
                            pred[i] = np.sum(r * w) / sw if abs(sw) > 1e-12 else self.fill
        return pred


# ------------------------- single-pair PCBC (structurally identical to exp1.PCBC, gate switchable) -------------------------
class PCBC_Gated(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=ALPHA, seed=42, gate=True, use_bary=False):
        self.alpha = alpha
        self.seed = seed
        self.gate = gate
        self.use_bary = use_bary
        self.lr = None
        self.pair = None
        self.inter = None

    def fit(self, X, y, year_id=None, pair=None):
        if year_id is None:
            year_id = np.ones(len(y))
        self.lr = LogisticRegression(class_weight="balanced", max_iter=500,
                                     random_state=self.seed)
        self.lr.fit(X, y)
        resid = y - self.lr.predict_proba(X)[:, 1]
        if pair is None:
            pair = exp1.select_best_pair_nested(
                X, y, self.lr.coef_[0], top_k=TOP_K, seed=42)
        self.pair = pair
        self.inter = LSIMGate(pair, gate=self.gate, use_bary=self.use_bary,
                              seed=self.seed)
        self.inter.fit(X, resid, y, year_id)
        return self

    def decision_function(self, X):
        return self.lr.decision_function(X) + self.alpha * self.inter.predict(X)

    def predict_proba(self, X):
        p = exp1.logit_to_prob(self.decision_function(X))
        return np.vstack([1 - p, p]).T


# ------------------------- temporal rolling smoke test (Polish Y1..Y5) -------------------------
def hull_out_rate(inter, X_te_norm):
    """Share of test points outside the convex hull (gate trigger rate)."""
    xy = X_te_norm[:, inter.pair]
    tri_idx = inter.tri.find_simplex(xy)
    return float(np.mean(tri_idx == -1))


def precision_at_k(prob, y, minority_label, k):
    idx = np.argsort(-prob)[:k]
    return float(np.mean(y[idx] == minority_label))


def eval_task_gated(task_id, train_ds_list, test_ds_name, n_repeat=N_REPEAT,
                    ood_shift=0.0):
    X_tr, y_tr = [], []
    for ds in train_ds_list:
        X_d, y_d, _, _ = exp1.load_dataset(ds)
        X_tr.append(X_d); y_tr.append(y_d)
    X_train = np.vstack(X_tr); y_train = np.hstack(y_tr)
    X_test, y_test, minority_label, _ = exp1.load_dataset(test_ds_name)
    mu = X_train.mean(0); sg = X_train.std(0) + 1e-6
    X_tr_n = (X_train - mu) / sg
    X_te_n = (X_test - mu) / sg
    if ood_shift > 0:
        # artificial OOD: shift every test feature (standardised units), moving the point cloud out of the training hull
        X_te_n = X_te_n + ood_shift

    pair = exp1.select_best_pair_nested(
        X_tr_n, y_train,
        LogisticRegression(max_iter=500, class_weight="balanced").fit(X_tr_n, y_train).coef_[0],
        top_k=TOP_K, seed=42)

    rows = []
    for seed in range(n_repeat):
        for gname, gate in [("PCBC-Gated", True), ("PCBC-NoGate", False)]:
            m = PCBC_Gated(seed=seed, gate=gate)
            m.fit(X_tr_n, y_train, pair=pair)
            prob = m.predict_proba(X_te_n)[:, 1]
            yp = (prob >= 0.5).astype(int)
            prf = precision_recall_fscore_support(y_test, yp, average=None, zero_division=0)
            uniq = np.unique(y_test)
            idx_min = 0 if uniq[0] == minority_label else 1
            rows.append({
                "task": task_id, "model": gname, "seed": seed,
                "P": prf[0][idx_min], "R": prf[1][idx_min], "F1": prf[2][idx_min],
                "AUC": roc_auc_score(y_test, prob),
                "P@10": precision_at_k(prob, y_test, minority_label, 10),
                "P@50": precision_at_k(prob, y_test, minority_label, 50),
                "hull_out_rate": hull_out_rate(m.inter, X_te_n),
            })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    year_ds = [f"polish_Bankruptcy{k}" for k in range(1, 6)]   # Y1..Y5
    frames = []
    for task_id in range(1, 5):
        train_ds = year_ds[:task_id]
        test_ds = year_ds[task_id]                              # Y{k+1}
        print(f"[task {task_id}] train {'+'.join(train_ds)} -> test {test_ds}",
              flush=True)
        frames.append(eval_task_gated(task_id, train_ds, test_ds))
    # extreme temporal gap: train on Y1 only -> test on Y5 (small training hull plus four years of drift, the gate should activate)
    print("[task 5] train polish_Bankruptcy1 -> test polish_Bankruptcy5 (extreme drift)",
          flush=True)
    frames.append(eval_task_gated(5, year_ds[:1], year_ds[4]))
    df = pd.concat(frames, ignore_index=True)
    agg = df.groupby(["task", "model"])[["P", "R", "F1", "AUC", "P@10", "P@50",
                                         "hull_out_rate"]].mean().round(4).reset_index()
    print("\n===== Hard-Boundary Ablation: Gated vs NoGate (polish temporal) =====",
          flush=True)
    print(agg.to_string(index=False), flush=True)
    # per-task gate trigger rate shown separately (geometric, independent of the gate, identical for both models)
    print("\n===== hull-out trigger rate by task (geometric, identical for both) =====",
          flush=True)
    print(agg.pivot_table(index="task", columns="model", values="hull_out_rate")["PCBC-Gated"]
          .rename("hull_out_rate").round(4).to_string(), flush=True)

    # artificial OOD sweep: task 5 fixed (train Y1 -> test Y5), all test features shifted
    print("\n===== Artificial OOD shift scan (train Y1 -> test Y5) =====", flush=True)
    ood_frames = []
    for shift in [0.0, 0.5, 1.0, 1.5, 2.0]:
        f = eval_task_gated(5, year_ds[:1], year_ds[4], ood_shift=shift)
        f["shift"] = shift
        ood_frames.append(f)
    ood = pd.concat(ood_frames, ignore_index=True)
    ood_agg = ood.groupby(["shift", "model"])[["P", "R", "F1", "AUC", "P@10", "P@50",
                                               "hull_out_rate"]].mean().round(4).reset_index()
    print(ood_agg.to_string(index=False), flush=True)
    print("\n===== hull-out rate by shift =====", flush=True)
    print(ood_agg.pivot_table(index="shift", columns="model", values="hull_out_rate")["PCBC-Gated"]
          .rename("hull_out_rate").round(4).to_string(), flush=True)
    df.to_csv(RESULT_CSV, index=False)
    ood.to_csv(os.path.join(DATA_DIR, "pcbc_nogate_ood_scan.csv"), index=False)
    print(f"\n[Saved] {RESULT_CSV} + pcbc_nogate_ood_scan.csv | total {time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
