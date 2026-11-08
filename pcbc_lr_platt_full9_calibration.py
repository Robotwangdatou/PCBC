# -*- coding: utf-8 -*-
"""
LR+Platt full-9 calibration diagnostic (paired protocol)
=========================================================
Protocol identical to pcbc_calibration_diagnostic.py (nine datasets, 5-fold, seed=42,
training-fold scaling, balanced LR). Added arm: LR+Platt (Platt scaling fitted on the training-fold
logits and applied to the test fold), which answers the reviewer question whether the calibration
improvement of PCBC (hier) is merely a Platt-scaling effect.
Metrics: ECE_10 (10 fixed bins, sample-size weighted), Brier, AUC (recorded as well).
Output: pcbc_lr_platt_full9_calibration.csv plus a grouped summary printout.
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import importlib.util

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_lr_platt_full9_calibration.csv")

DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]


def ece_bins(p, y, bins=10):
    """Expected calibration error with 10 fixed bins (same convention as pcbc_calibration_diagnostic.py)."""
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


def main():
    t_all = time.time()
    parts = []
    for ds in DATASETS:
        t0 = time.time()
        X, y, minority_label, min_ratio = exp1.load_dataset(ds)
        n = len(y)
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        rows = []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
            ytr, yte = y[tr], y[te]
            lr = LogisticRegression(max_iter=500, class_weight="balanced",
                                    random_state=42)
            lr.fit(Xtr, ytr)
            logits_tr = lr.decision_function(Xtr)
            logits_te = lr.decision_function(Xte)
            platt = LogisticRegression(C=1.0, max_iter=500)
            platt.fit(logits_tr.reshape(-1, 1), ytr)
            prob = platt.predict_proba(logits_te.reshape(-1, 1))[:, 1]
            rows.append({
                "dataset": ds, "model": "LR+Platt", "fold": fold,
                "ece10": ece_bins(prob, yte),
                "brier": brier_score_loss(yte, prob),
                "auc": roc_auc_score(yte, prob),
            })
        df = pd.DataFrame(rows)
        summ = df.groupby("model").agg(
            ece10_mean=("ece10", "mean"), ece10_std=("ece10", "std"),
            brier_mean=("brier", "mean"), brier_std=("brier", "std"),
            auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ).reset_index()
        summ.insert(0, "dataset", ds)
        summ.insert(1, "n_samples", n)
        summ["test_minor_ratio"] = min_ratio
        summ["elapsed_s"] = round(time.time() - t0, 1)
        parts.append(summ)
        print(f"[{ds} done] n={n} elapsed={summ['elapsed_s'].iloc[0]}s",
              flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RESULT_CSV, index=False)

    imp6 = ["US_Bankruptcy", "bank-full", "give me some credit",
            "polish_Bankruptcy5", "taiwan_bankruptcy", "taiwan_credit"]
    print("\n===== LR+Platt grouped means (4dp) =====", flush=True)
    for name, sub in [("All", out), ("Imp", out[out.dataset.isin(imp6)]),
                      ("Bal", out[~out.dataset.isin(imp6)])]:
        m = sub.groupby("dataset")[["ece10_mean", "brier_mean"]].mean().mean()
        print(f"{name}: ECE {m['ece10_mean']:.4f}  Brier {m['brier_mean']:.4f}",
              flush=True)
    print(f"\n[Saved] {RESULT_CSV} | total {time.time() - t_all:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
