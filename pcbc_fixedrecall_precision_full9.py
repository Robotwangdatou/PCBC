# -*- coding: utf-8 -*-
"""
Fixed-Recall Precision comparison (9 datasets x 5 folds, protocol-consistent)
=============================================================================
Compares the minority Precision of every model at the fixed operating points Recall = 0.5 / 0.6 / 0.7.
Standard PR-curve evaluation: sweep the threshold on each fold's test set and interpolate linearly to
Precision@Recall=target (the model is already fitted and the threshold is an evaluation knob, so there is no leakage).
Models: LR / PCBC / PCBC-MS-H3 / EBM / DT / StrictPLTR (the full family of Table 2).
Output: pcbc_fixedrecall_precision_full9.csv
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import importlib.util

warnings.filterwarnings("ignore")

_spec1 = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec1)
_spec1.loader.exec_module(exp1)
_spec4 = importlib.util.spec_from_file_location(
    "exp4", os.path.join(_CODE_DIR, "pcbc_multisubspace_experiment.py"))
exp4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(exp4)
_spec_all = importlib.util.spec_from_file_location(
    "allm", os.path.join(_CODE_DIR, "pcbc_allmodels_full9.py"))
allm = importlib.util.module_from_spec(_spec_all)
_spec_all.loader.exec_module(allm)
_spec_sp = importlib.util.spec_from_file_location(
    "dtp", os.path.join(_CODE_DIR, "pcbc_dt_pltr_full9.py"))
dtp = importlib.util.module_from_spec(_spec_sp)
_spec_sp.loader.exec_module(dtp)
StrictPLTR = dtp.StrictPLTR

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]
TARGETS = [0.5, 0.6, 0.7]


def prec_at_recall(p, y, target):
    """Linear-interpolated Precision at a target Recall (minority = 1)."""
    if p.min() >= p.max():
        return float("nan")
    order = np.argsort(p)[::-1]
    ps, ys = p[order], y[order]
    cum_tp = np.cumsum(ys)
    cum_rec = cum_tp / max(ys.sum(), 1)
    cum_prec = cum_tp / np.arange(1, len(ys) + 1)
    # find first index with cum_rec >= target
    idx = np.searchsorted(cum_rec, target, side="left")
    if idx == 0:
        return float(cum_prec[0])
    if idx >= len(ys):
        return float(cum_prec[-1])
    # linear interpolation between idx-1 and idx
    r0, r1 = cum_rec[idx - 1], cum_rec[idx]
    if r1 <= r0:
        return float(cum_prec[idx])
    w = (target - r0) / (r1 - r0)
    return float(cum_prec[idx - 1] + w * (cum_prec[idx] - cum_prec[idx - 1]))


def run_dataset(ds_name):
    t0 = time.time()
    X, y, minority_label, min_ratio = exp1.load_dataset(ds_name)
    n = len(y)
    exp4.TOP_K = 20 if n >= 30000 else 30
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        ytr, yte = y[tr], y[te]
        ymin_te = (yte == minority_label).astype(int)

        lr = LogisticRegression(class_weight="balanced", max_iter=500)
        lr.fit(Xtr, ytr)
        dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                    random_state=42)
        dt.fit(Xtr, ytr)
        ebm = allm.EBMHandcrafted(n_bins=64, n_rounds=20, learning_rate=0.05)
        ebm.fit(Xtr, ytr)
        sp = StrictPLTR(n_base_trees=20, max_tree_depth=2,
                        reg_strength=0.01, random_state=42)
        sp.fit(Xtr, ytr)
        m_pcbc = exp4.PCBC_MS(seed=42, K=1, mode="parallel")
        m_pcbc.fit(Xtr, ytr)
        m_ms = exp4.PCBC_MS(seed=42, K=3, mode="hier")
        m_ms.fit(Xtr, ytr)

        probs = {
            "LR": lr.predict_proba(Xte)[:, 1],
            "DT": dt.predict_proba(Xte)[:, 1],
            "StrictPLTR": sp.predict_proba(Xte)[:, 1],
            "EBM": ebm.predict_proba(Xte)[:, 1],
            "PCBC": m_pcbc.predict_proba(Xte)[:, 1],
            "PCBC-MS-H3": m_ms.predict_proba(Xte)[:, 1],
        }
        for model, p in probs.items():
            for tgt in TARGETS:
                rows.append({
                    "dataset": ds_name, "fold": fold, "model": model,
                    "recall_target": tgt,
                    "precision": prec_at_recall(p, ymin_te, tgt),
                })
        print(f"  [fold {fold} done] {ds_name}", flush=True)
    df = pd.DataFrame(rows)
    print(f"[{ds_name} done] elapsed={time.time()-t0:.0f}s", flush=True)
    return df


def main():
    t_all = time.time()
    parts = [run_dataset(ds) for ds in DATASETS]
    df = pd.concat(parts, ignore_index=True)
    out = os.path.join(DATA_DIR, "pcbc_fixedrecall_precision_full9.csv")
    df.to_csv(out, index=False)

    g = df.groupby(["model", "recall_target"])["precision"].mean().round(4)
    print("\n===== Precision @ fixed Recall (9-dataset mean) =====")
    print(g.unstack().to_string())
    print("\n===== paired diff vs PCBC-MS-H3 (dataset-level Wilcoxon) =====")
    from scipy.stats import wilcoxon
    piv = df.groupby(["model", "recall_target", "dataset"])["precision"].mean().reset_index()
    ms = piv[piv.model == "PCBC-MS-H3"].set_index(["recall_target", "dataset"])["precision"]
    for tgt in TARGETS:
        line = f"Recall={tgt}: "
        for m in ["LR", "DT", "StrictPLTR", "EBM", "PCBC"]:
            o = piv[piv.model == m].set_index(["recall_target", "dataset"])["precision"]
            j = pd.concat([ms, o], axis=1, join="inner").dropna()
            d = j.iloc[:, 0] - j.iloc[:, 1]
            pos = (d > 0).sum()
            try:
                _, p = wilcoxon(d[d != 0])
            except ValueError:
                p = 2 * (0.5 ** len(d))
            line += f"{m} {pos}/{len(d)} p={p:.4f} | "
        print(line)
    print(f"\n[Saved] {out} | total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
