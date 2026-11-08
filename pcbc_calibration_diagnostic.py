# -*- coding: utf-8 -*-
"""
Calibration Diagnostic: PCBC vs MS-H3 vs LR on 9 datasets
=========================================================
Purpose: answer whether the probability output of PCBC (hier) is well calibrated - i.e. whether an explicit calibration step (isotonic/Platt) belongs in the paper.
Protocol identical to pcbc_msh3_full9.py (nine datasets, 5-fold, seed=42, top_k 12/20/30 by sample size).
Metrics:
  ECE_10 : expected calibration error (10 fixed bins, mean confidence versus mean accuracy, sample-size weighted)
  Brier  : Brier score（mean((p-y)^2)）
  AUC    : recorded as well (cross-checked against full9 to confirm the protocol matches)
Models:
  LR        : class_weight=balanced (linear reference, usually well calibrated)
  PCBC      : original single-subspace variant
  PCBC-MS-H3: hierarchical serial re-centering, K=3
Output: pcbc_calibration_diagnostic.csv (ECE/Brier/AUC mean +/- std per dataset and model)
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, brier_score_loss)
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
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_calibration_diagnostic.csv")

DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]


def effective_topk(n):
    if n >= 100000:
        return 12
    if n >= 30000:
        return 20
    return 30


def ece_bins(p, y, bins=10):
    """Expected calibration error with 10 fixed bins (mean confidence versus observed frequency, sample-size weighted)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    edges[-1] += 1e-12
    idx = np.digitize(p, edges, right=False) - 1
    idx = np.clip(idx, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        ece += (m.sum() / len(y)) * abs(conf - acc)
    return ece


def run_dataset(ds_name):
    t0 = time.time()
    X, y, minority_label, min_ratio = exp1.load_dataset(ds_name)
    n = len(y)
    topk = effective_topk(n)
    exp4.TOP_K = topk
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    specs = {"LR": None, "PCBC": dict(K=1, mode="parallel"),
             "PCBC-MS-H3": dict(K=3, mode="hier")}
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        ytr, yte = y[tr], y[te]
        # LR
        lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        lr.fit(Xtr, ytr)
        prob_lr = lr.predict_proba(Xte)[:, 1]
        # PCBC / MS-H3
        prob_pcbc = None
        prob_h3 = None
        for vname, kw in specs.items():
            if vname == "LR":
                continue
            m = exp4.PCBC_MS(seed=42, **kw)
            m.fit(Xtr, ytr)
            prob = m.predict_proba(Xte)[:, 1]
            if vname == "PCBC":
                prob_pcbc = prob
            else:
                prob_h3 = prob
        for vname, prob in [("LR", prob_lr), ("PCBC", prob_pcbc),
                            ("PCBC-MS-H3", prob_h3)]:
            rows.append({
                "dataset": ds_name, "model": vname, "fold": fold,
                "ece10": ece_bins(prob, yte),
                "brier": brier_score_loss(yte, prob),
                "auc": roc_auc_score(yte, prob),
            })
        print(f"  [fold {fold} done] {ds_name}", flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("model").agg(
        ece10_mean=("ece10", "mean"), ece10_std=("ece10", "std"),
        brier_mean=("brier", "mean"), brier_std=("brier", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
    ).reset_index()
    summ.insert(0, "dataset", ds_name)
    summ.insert(1, "n_samples", n)
    summ["test_minor_ratio"] = min_ratio
    summ["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[{ds_name} done] n={n} elapsed={summ['elapsed_s'].iloc[0]}s", flush=True)
    return summ


def main():
    t_all = time.time()
    parts = []
    for ds in DATASETS:
        parts.append(run_dataset(ds))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RESULT_CSV, index=False)
    print(out.round(4).to_string(index=False), flush=True)
    # nine-dataset mean summary (cross-model comparison)
    print("\n===== 9-dataset means =====", flush=True)
    agg = out.groupby("model").agg(
        ece10=("ece10_mean", "mean"),
        brier=("brier_mean", "mean"),
        auc=("auc_mean", "mean"),
    ).reset_index()
    print(agg.round(4).to_string(index=False), flush=True)
    print(f"\n[Saved] {RESULT_CSV}  |  total {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
