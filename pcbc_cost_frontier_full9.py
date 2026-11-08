# -*- coding: utf-8 -*-
"""
Cost-Frontier Formal Experiment: MS-H3 vs baselines under audit cost (full 9 datasets)
=======================================================================================
Turns the cost-curve audit narrative into a formal experiment:
  audit cost = r * FP + FN   (r = review cost of a false positive / loss cost of a false negative, r in {0.25,0.5,1,2,4})
  threshold fixed at 0.5; normalised cost = cost / n_test

Reference models (same 5-fold protocol, seed=42, same convention as pcbc_msh3_full9):
  LR            : uncorrected baseline (balanced), the no-correction reference of the cost frontier
  PCBC          : original single-subspace variant (K=1, alpha=0.6)
  PCBC-MS-H3    : hierarchical serial re-centering (K=3, alpha=0.6)

Significance: Wilcoxon signed-rank on the paired cost
  - fold level n=45 (nine datasets x five folds)
  - dataset level n=9 (five-fold mean per dataset, the conservative convention)

Output:
  pcbc_cost_frontier_full9_results.csv   one row per fold (cost at every r)
  pcbc_cost_frontier_full9_summary.csv   mean cost per dataset x model x r
  pcbc_cost_frontier_full9_wilcoxon.csv  paired-test results

Run: python pcbc_cost_frontier_full9.py
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support)
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
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_cost_frontier_full9_results.csv")
SUMMARY_CSV = os.path.join(DATA_DIR, "pcbc_cost_frontier_full9_summary.csv")
WILCOX_CSV = os.path.join(DATA_DIR, "pcbc_cost_frontier_full9_wilcoxon.csv")

DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]
R_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
ALPHA = 0.6


def effective_topk(n):
    if n >= 100000:
        return 12
    if n >= 30000:
        return 20
    return 30


def audit_cost(prob, y, r):
    yp = (prob >= 0.5).astype(int)
    fp = int(np.sum((yp == 1) & (y == 0)))
    fn = int(np.sum((yp == 0) & (y == 1)))
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

        # ---- LR baseline (no correction) ----
        lr = LogisticRegression(max_iter=500, class_weight="balanced",
                                random_state=42)
        lr.fit(Xtr, ytr)
        prob_lr = lr.predict_proba(Xte)[:, 1]

        # ---- PCBC / PCBC-MS-H3 (same 5-fold protocol) ----
        m_pcbc = exp4.PCBC_MS(seed=42, K=1, mode="parallel")
        m_pcbc.fit(Xtr, ytr)
        prob_pcbc = m_pcbc.predict_proba(Xte)[:, 1]
        m_h3 = exp4.PCBC_MS(seed=42, K=3, mode="hier")
        m_h3.fit(Xtr, ytr)
        prob_h3 = m_h3.predict_proba(Xte)[:, 1]

        probs = {"LR": prob_lr, "PCBC": prob_pcbc, "PCBC-MS-H3": prob_h3}
        for vname, prob in probs.items():
            for r in R_GRID:
                cost, fp, fn = audit_cost(prob, yte, r)
                rows.append({
                    "dataset": ds_name, "fold": fold, "model": vname, "r": r,
                    "cost": cost / len(yte), "fp": fp, "fn": fn,
                    "auc": roc_auc_score(yte, prob),
                })
        # minority-class P/R/F1 (independent of r, recorded once in the r=1.0 row to avoid duplication)
        prf = precision_recall_fscore_support(
            yte, (prob_h3 >= 0.5).astype(int), average=None, zero_division=0)
        uniq = np.unique(yte)
        idx_min = 0 if uniq[0] == minority_label else 1
        rows.append({
            "dataset": ds_name, "fold": fold, "model": "MS-H3-metrics",
            "r": 1.0, "cost": np.nan, "fp": np.nan, "fn": np.nan,
            "auc": roc_auc_score(yte, prob_h3),
            "precision": prf[0][idx_min], "recall": prf[1][idx_min],
            "f1": prf[2][idx_min],
        })
        print(f"  [fold {fold} done] {ds_name}", flush=True)

    df = pd.DataFrame(rows)
    print(f"[{ds_name} done] n={n} min_ratio={min_ratio:.4f} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)
    return df


def main():
    t_all = time.time()
    parts = []
    for ds in DATASETS:
        parts.append(run_dataset(ds))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(RESULT_CSV, index=False)

    # ---- summary: mean cost per dataset x model x r ----
    cost_rows = df[df["cost"].notna()].copy()
    summ = cost_rows.groupby(["dataset", "model", "r"])["cost"].mean().reset_index()
    summ.to_csv(SUMMARY_CSV, index=False)

    # ---- cost frontier: nine-dataset mean (model x r) ----
    pivot = summ.groupby(["model", "r"])["cost"].mean().reset_index()
    pivot9 = pivot.pivot_table(index="model", columns="r", values="cost").round(4)
    print("\n===== Nine-dataset mean normalised audit cost (model x r) =====", flush=True)
    print(pivot9.to_string(), flush=True)

    # ---- Wilcoxon signed-rank (per r) ----
    w_rows = []
    for r in R_GRID:
        sub = cost_rows[cost_rows["r"] == r]
        for a, b in [("PCBC-MS-H3", "LR"), ("PCBC-MS-H3", "PCBC"),
                     ("PCBC", "LR")]:
            # fold-level pairing, n=45
            pa = sub[sub["model"] == a].set_index(["dataset", "fold"])["cost"]
            pb = sub[sub["model"] == b].set_index(["dataset", "fold"])["cost"]
            idx = pa.index.intersection(pb.index)
            da = pa.loc[idx].to_numpy(); db = pb.loc[idx].to_numpy()
            diff = da - db
            if np.all(diff == 0):
                w_f, p_f = np.nan, np.nan
            else:
                w_f, p_f = wilcoxon(da, db)
            # dataset-level pairing, n=9 (five-fold mean per dataset)
            ga = sub[sub["model"] == a].groupby("dataset")["cost"].mean()
            gb = sub[sub["model"] == b].groupby("dataset")["cost"].mean()
            gi = ga.index.intersection(gb.index)
            if np.all(ga.loc[gi].to_numpy() == gb.loc[gi].to_numpy()):
                w_g, p_g = np.nan, np.nan
            else:
                w_g, p_g = wilcoxon(ga.loc[gi].to_numpy(), gb.loc[gi].to_numpy())
            w_rows.append({
                "r": r, "A": a, "B": b,
                "A_cost_mean": float(da.mean()), "B_cost_mean": float(db.mean()),
                "diff_mean": float(diff.mean()),
                "n_fold": len(da), "W_fold": w_f, "p_fold": p_f,
                "n_dataset": len(gi), "W_dataset": w_g, "p_dataset": p_g,
                "direction": "A<B" if diff.mean() < 0 else "A>B",
            })
    wdf = pd.DataFrame(w_rows)
    wdf.to_csv(WILCOX_CSV, index=False)
    print("\n===== Wilcoxon signed-rank (paired cost) =====", flush=True)
    print(wdf.round(4).to_string(index=False), flush=True)
    print(f"\n[Saved] {RESULT_CSV} / {SUMMARY_CSV} / {WILCOX_CSV}  |  "
          f"total {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
