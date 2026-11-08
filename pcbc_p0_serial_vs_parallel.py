# -*- coding: utf-8 -*-
"""
P0 Experiment: Serial vs. Parallel PCBC (direct head-to-head)
============================================================
Direct experimental support for Proposition 1 of the paper (serial contraction vs parallel bias).

Reference models (same 5-fold protocol, seed=42, same convention as pcbc_msh3_full9 / pcbc_cost_frontier):
  PCBC (serial)   : hierarchical serial re-centering, K=3, each layer fits that layer's residual r^(k-1),
                    prediction accumulates the corrections  logit = s0 + alpha * sum_k delta^(k)(x)
  PCBC-Parallel   : K=3 parallel layers, each fitted independently on the INITIAL residual r^(0),
                    prediction adds the layer corrections directly  logit = s0 + alpha * sum_k delta_par^(k)(x)
                    - consistent with the mechanism E[alpha sum_k delta_par^(k)] >= E[alpha sum_k delta^(k)] of Proposition 1:
                    parallel layers fit the same initial residual and combine same-sign corrections, which theoretically overstates majority risk.

Reference baseline:
  LR              : uncorrected linear baseline (the reference for the cost frontier / FPR)

Reported metrics:
  1) Precision@10/20/30/50  : the top-K alerts by descending model probability, and the true minority share inside them
  2) audit-cost curve       : cost = r*FP + FN, r in {0.25, 0.5, 1, 2, 4}, threshold 0.5, normalised cost/n_test
  3) majority/minority FPR  : FPR_majority = FP_maj/N_maj, FPR_minority = FP_min/N_min (at the 0.5 threshold)
                             tests the claim that the parallel design overstates majority risk
  4) standard metrics       : P / R / F1 / AUC / PR-AUC (cross-checked against the existing results)
  5) significance           : Wilcoxon signed-rank (fold level n=45 and dataset level n=9)

Output:
  pcbc_p0_serial_vs_parallel_results.csv    per-fold detail
  pcbc_p0_serial_vs_parallel_summary.csv    mean per dataset x model x r
  pcbc_p0_serial_vs_parallel_wilcoxon.csv   paired tests (per metric and r)
  pcbc_p0_serial_vs_parallel_metrics.csv    standard metrics + Precision@K + FPR per dataset and model

Run: python pcbc_p0_serial_vs_parallel.py
      (requires scipy / sklearn / pandas / numpy, same environment as the existing experiments)
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
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from scipy.stats import wilcoxon
import importlib.util

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)
_spec4 = importlib.util.spec_from_file_location(
    "exp4", os.path.join(_CODE_DIR, "pcbc_multisubspace_experiment.py"))
exp4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(exp4)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(DATA_DIR, "pcbc_p0_serial_vs_parallel_results.csv")
SUMMARY_CSV = os.path.join(DATA_DIR, "pcbc_p0_serial_vs_parallel_summary.csv")
WILCOX_CSV = os.path.join(DATA_DIR, "pcbc_p0_serial_vs_parallel_wilcoxon.csv")
METRICS_CSV = os.path.join(DATA_DIR, "pcbc_p0_serial_vs_parallel_metrics.csv")

DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]
R_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
PREC_K = [10, 20, 30, 50]
ALPHA = 0.6
K_LAYERS = 3


def effective_topk(n):
    if n >= 100000:
        return 12
    if n >= 30000:
        return 20
    return 30


# ---------------------------------------------------------------------------
# PCBC-Parallel: each layer fits the initial residual r^(0) independently; predictions add the layer corrections (sum)
# difference from exp4.PCBC_MS(mode='parallel'):
#   exp4 parallel = each layer's correction is averaged (mean) -> equivalent to treating the ensemble as bagging;
#   this P0 parallel = each layer's correction is summed (sum) -> the mechanism defined in Proposition 1:
#   parallel layers fit the same initial residual, same-sign corrections compound, exposing the overstatement of majority risk.
# ---------------------------------------------------------------------------
class PCBC_ParallelSum(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=ALPHA, seed=42, K=K_LAYERS, use_bary=True):
        self.alpha = alpha
        self.seed = seed
        self.K = K
        self.use_bary = use_bary
        self.lr = None
        self.layers = None

    def fit(self, X, y, pairs=None):
        self.lr = LogisticRegression(max_iter=500, class_weight="balanced",
                                     random_state=self.seed)
        self.lr.fit(X, y)
        if pairs is None:
            pairs, _ = exp4.select_top_k_pairs(X, y, self.lr.coef_[0],
                                               top_k=exp4.TOP_K, K=self.K,
                                               seed=self.seed)
        self.pairs = pairs
        # each layer fits the INITIAL residual r^(0) = y - p^(0) independently (parallel definition: no layerwise update)
        resid0 = y - self.lr.predict_proba(X)[:, 1]
        self.layers = []
        for p in pairs:
            sim = exp1.TemporalLSIM(pair=p, use_bary=self.use_bary, seed=self.seed)
            sim.fit(X, resid0, y, np.ones(len(y)))
            self.layers.append(sim)
        return self

    def decision_function(self, X):
        logit = self.lr.decision_function(X)
        for sim in self.layers:
            logit += self.alpha * sim.predict(X)   # layer corrections added directly (sum)
        return logit

    def predict_proba(self, X):
        p = exp1.logit_to_prob(self.decision_function(X))
        return np.vstack([1 - p, p]).T


# ---------------------------------------------------------------------------
# metric utilities
# ---------------------------------------------------------------------------
def precision_at_k(prob, y, minority_label, K):
    """Take the top-K alerts by descending probability and return the true minority share among them."""
    top = np.argsort(-prob)[:K]
    return float(np.mean(y[top] == minority_label))


def fpr_split(prob, y, minority_label):
    """Majority/minority FPR at the 0.5 threshold, computed directly on the raw labels."""
    yp = (prob >= 0.5).astype(int)
    y_min = (y == minority_label).astype(int)
    # majority = not minority
    maj_mask = ~y_min.astype(bool)
    min_mask = y_min.astype(bool)
    fp_maj = int(np.sum((yp[maj_mask] == 1)))      # majority records predicted positive
    fp_min = int(np.sum((yp[min_mask] == 1)))      # minority records predicted positive (the TPR side)
    n_maj = int(maj_mask.sum())
    n_min = int(min_mask.sum())
    return (fp_maj / n_maj if n_maj else np.nan,
            fp_min / n_min if n_min else np.nan)


def audit_cost(prob, y, r, minority_label):
    yp = (prob >= 0.5).astype(int)
    y_min = (y == minority_label).astype(int)
    fp = int(np.sum((yp == 1) & (y_min == 0)))
    fn = int(np.sum((yp == 0) & (y_min == 1)))
    return r * fp + fn, fp, fn


def run_dataset(ds_name):
    t0 = time.time()
    X, y, minority_label, min_ratio = exp1.load_dataset(ds_name)
    n = len(y)
    topk = effective_topk(n)
    exp4.TOP_K = topk
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        ytr, yte = y[tr], y[te]

        # LR baseline
        lr = LogisticRegression(max_iter=500, class_weight="balanced",
                                random_state=42)
        lr.fit(Xtr, ytr)
        prob_lr = lr.predict_proba(Xte)[:, 1]

        # serial PCBC (K=3)
        m_ser = exp4.PCBC_MS(seed=42, K=K_LAYERS, mode="hier")
        m_ser.fit(Xtr, ytr)
        prob_ser = m_ser.predict_proba(Xte)[:, 1]

        # parallel PCBC-Parallel (K=3, summed)
        m_par = PCBC_ParallelSum(seed=42, K=K_LAYERS)
        m_par.fit(Xtr, ytr)
        prob_par = m_par.predict_proba(Xte)[:, 1]

        probs = {"LR": prob_lr, "PCBC-serial": prob_ser, "PCBC-Parallel": prob_par}
        for vname, prob in probs.items():
            for r in R_GRID:
                cost, fp, fn = audit_cost(prob, yte, r, minority_label)
                fpr_maj, fpr_min = fpr_split(prob, yte, minority_label)
                prf = precision_recall_fscore_support(
                    yte, (prob >= 0.5).astype(int), average=None, zero_division=0)
                uniq = np.unique(yte)
                idx_min = 0 if uniq[0] == minority_label else 1
                row = {
                    "dataset": ds_name, "fold": fold, "model": vname, "r": r,
                    "cost": cost / len(yte), "fp": fp, "fn": fn,
                    "fpr_majority": fpr_maj, "fpr_minority": fpr_min,
                    "precision": prf[0][idx_min], "recall": prf[1][idx_min],
                    "f1": prf[2][idx_min],
                    "auc": roc_auc_score(yte, prob),
                    "auc_pr": average_precision_score(yte, prob),
                }
                for K in PREC_K:
                    row[f"prec_at_{K}"] = precision_at_k(prob, yte, minority_label, K)
                rows.append(row)
        print(f"  [fold {fold} done] {ds_name}", flush=True)

    df = pd.DataFrame(rows)
    print(f"[{ds_name} done] n={n} min_ratio={min_ratio:.4f} topk={topk} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)
    return df


def _plot(summary_df, met9_df):
    """Draw two figures: (1) cost curve, (2) Precision@K, in the same directory as the CSV."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped] matplotlib unavailable: {e}", flush=True)
        return
    fig_dir = os.path.join(DATA_DIR, "p0_figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- figure 1: audit-cost curve (nine-dataset mean) ----
    pv = summary_df.groupby(["model", "r"])["cost"].mean().reset_index()
    pvt = pv.pivot_table(index="r", columns="model", values="cost").sort_index()
    plt.figure(figsize=(7, 4.5))
    for col in pvt.columns:
        plt.plot(pvt.index, pvt[col], marker="o", label=col, linewidth=1.8)
    plt.xlabel("Cost ratio $r$ (FP / FN)")
    plt.ylabel("Normalized audit cost (mean over 9 datasets)")
    plt.xscale("log", base=2)
    plt.title("Audit-cost frontier: serial vs. parallel PCBC")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "cost_frontier_serial_vs_parallel.png")
    plt.savefig(p1, dpi=200)
    plt.close()

    # ---- figure 2: Precision@K (minority top-K precision) ----
    pk_cols = [c for c in met9_df.columns if c.startswith("prec_at_")]
    ks = [int(c.split("_")[-1]) for c in pk_cols]
    plt.figure(figsize=(7, 4.5))
    for model in met9_df.index:
        plt.plot(ks, met9_df.loc[model, pk_cols].values, marker="o",
                 label=model, linewidth=1.8)
    plt.xlabel("Top-K alerts reviewed")
    plt.ylabel("Precision@K (minority fraction among top K)")
    plt.title("Precision@K: serial vs. parallel PCBC")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    p2 = os.path.join(fig_dir, "precision_at_k_serial_vs_parallel.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"[plots saved] {p1}\n               {p2}", flush=True)


def main():
    t_all = time.time()
    parts = []
    for ds in DATASETS:
        parts.append(run_dataset(ds))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)

    # ---- summary: mean cost per dataset x model x r ----
    cost_rows = df[df["cost"].notna()].copy()
    summ = cost_rows.groupby(["dataset", "model", "r"])["cost"].mean().reset_index()
    summ.to_csv(SUMMARY_CSV, index=False)
    pivot = summ.groupby(["model", "r"])["cost"].mean().reset_index()
    pivot9 = pivot.pivot_table(index="model", columns="r", values="cost").round(4)
    print("\n===== Nine-dataset mean normalised audit cost (model x r) =====", flush=True)
    print(pivot9.to_string(), flush=True)

    # ---- dataset-level means of the standard metrics / Precision@K / FPR ----
    metrics_cols = ["precision", "recall", "f1", "auc", "auc_pr",
                    "fpr_majority", "fpr_minority"] + [f"prec_at_{K}" for K in PREC_K]
    met = cost_rows.groupby(["dataset", "model"])[metrics_cols].mean().reset_index()
    met.to_csv(METRICS_CSV, index=False)
    met9 = met.groupby("model")[metrics_cols].mean().round(4)
    print("\n===== Nine-dataset mean: standard metrics + FPR + Precision@K =====", flush=True)
    print(met9.to_string(), flush=True)

    # ---- Wilcoxon signed-rank (serial versus parallel, per metric and r) ----
    w_rows = []
    # cost + FPR (split by r)
    for r in R_GRID:
        sub = cost_rows[cost_rows["r"] == r]
        for metric in ["cost", "fpr_majority", "fpr_minority"]:
            w_rows += _wilcox_pair(sub, metric, r)
    # standard metrics + Precision@K (independent of r, taken from the r=1.0 row)
    sub = cost_rows[cost_rows["r"] == 1.0]
    for metric in metrics_cols:
        w_rows += _wilcox_pair(sub, metric, None)
    wdf = pd.DataFrame(w_rows)
    wdf.to_csv(WILCOX_CSV, index=False)
    print("\n===== Wilcoxon signed-rank（PCBC-Parallel vs PCBC-serial） =====", flush=True)
    print(wdf.round(4).to_string(index=False), flush=True)
    print(f"\n[Saved] {RESULTS_CSV} / {SUMMARY_CSV} / {METRICS_CSV} / {WILCOX_CSV}  |  "
          f"total {time.time()-t_all:.1f}s", flush=True)

    _plot(summ, met9)


def _wilcox_pair(sub, metric, r):
    """Paired test of serial versus parallel: fold level n=45 and dataset level n=9. Returns list[dict]."""
    out = []
    for a, b in [("PCBC-Parallel", "PCBC-serial")]:
        pa = sub[sub["model"] == a].set_index(["dataset", "fold"])[metric]
        pb = sub[sub["model"] == b].set_index(["dataset", "fold"])[metric]
        idx = pa.index.intersection(pb.index)
        da = pa.loc[idx].to_numpy(); db = pb.loc[idx].to_numpy()
        diff = da - db
        if np.all(diff == 0):
            w_f, p_f = np.nan, np.nan
        else:
            w_f, p_f = wilcoxon(da, db)
        ga = sub[sub["model"] == a].groupby("dataset")[metric].mean()
        gb = sub[sub["model"] == b].groupby("dataset")[metric].mean()
        gi = ga.index.intersection(gb.index)
        if np.all(ga.loc[gi].to_numpy() == gb.loc[gi].to_numpy()):
            w_g, p_g = np.nan, np.nan
        else:
            w_g, p_g = wilcoxon(ga.loc[gi].to_numpy(), gb.loc[gi].to_numpy())
        out.append({
            "metric": metric, "r": r, "A": a, "B": b,
            "A_mean": float(da.mean()), "B_mean": float(db.mean()),
            "diff_mean": float(diff.mean()),
            "n_fold": len(da), "W_fold": w_f, "p_fold": p_f,
            "n_dataset": len(gi), "W_dataset": w_g, "p_dataset": p_g,
            "direction": "A>B" if diff.mean() > 0 else ("A<B" if diff.mean() < 0 else "="),
        })
    return out


if __name__ == "__main__":
    main()
