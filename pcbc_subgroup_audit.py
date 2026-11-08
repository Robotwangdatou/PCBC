# -*- coding: utf-8 -*-
"""
Subgroup Audit: PCBC-MS-H3 vs LR by feature-quantile subgroups
==============================================================
Purpose: population-level diagnostic for the auditable claim - an auditor inspects model behaviour
by semantically meaningful feature quantile groups. Protocol identical to the main experiment (StratifiedKFold(5, seed=42), training-fold
z-score、LR balanced、PCBC-MS-H3 (K=3, alpha=0.6, mode='hier')）。

Audit features (regulatory / risk semantics):
  - german               : col 12 = age (age, a common risk dimension in ECOA contexts)
  - give me some credit  : col 3  = DebtRatio (debt ratio, a high-risk driver)

Per fold, training-fold tercile boundaries split the test fold into low / mid / high groups; reported:
  n, minority ratio, LR Precision, PCBC (hier) Precision, PCBC (hier) ECE, PCBC (hier) Brier,
  evidence coverage (share of queries inside a simplex at all three layers / at least one layer)

Output: pcbc_subgroup_audit.csv (per dataset, fold and group) plus the aggregate
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, precision_score
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

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_subgroup_audit.csv")

# dataset -> (audit feature index, display name)
AUDIT_FEATURES = {
    "german": (12, "Age"),
    "give me some credit": (3, "DebtRatio"),
}


def ece_bins(p, y, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    edges[-1] += 1e-12
    idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return ece


def group_indices(feat_train, feat_test):
    """Training-fold tercile boundaries -> low/mid/high indices of the test fold"""
    q1, q2 = np.quantile(feat_train, [1.0 / 3, 2.0 / 3])
    low = np.nonzero(feat_test <= q1)[0]
    mid = np.nonzero((feat_test > q1) & (feat_test <= q2))[0]
    high = np.nonzero(feat_test > q2)[0]
    return {"low": low, "mid": mid, "high": high}


def run_dataset(ds_name):
    t0 = time.time()
    feat_idx, feat_name = AUDIT_FEATURES[ds_name]
    X, y, minority_label, min_ratio = exp1.load_dataset(ds_name)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        ytr, yte = y[tr], y[te]
        # audit feature on the raw scale (groups use raw values, quantile edges taken from the training-fold raw distribution)
        feat_tr_raw = X[tr][:, feat_idx]
        feat_te_raw = X[te][:, feat_idx]
        # models
        lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        lr.fit(Xtr, ytr)
        prob_lr = lr.predict_proba(Xte)[:, 1]
        m = exp4.PCBC_MS(seed=42, K=3, mode="hier")
        m.fit(Xtr, ytr)
        prob_h3 = m.predict_proba(Xte)[:, 1]
        # evidence coverage: simplex hits per layer
        layer_hit = []
        for sim in m.layers:
            q = sim.tri.find_simplex(Xte[:, sim.pair])
            layer_hit.append(q != -1)
        layer_hit = np.vstack(layer_hit)          # (L, n_te)
        all_hit = layer_hit.all(axis=0)
        any_hit = layer_hit.any(axis=0)
        # group statistics
        for gname, gidx in group_indices(feat_tr_raw, feat_te_raw).items():
            if len(gidx) == 0:
                continue
            n = len(gidx)
            mr = yte[gidx].mean()
            p_lr = precision_score(yte[gidx], (prob_lr[gidx] >= 0.5).astype(int),
                                   zero_division=0)
            p_h3 = precision_score(yte[gidx], (prob_h3[gidx] >= 0.5).astype(int),
                                   zero_division=0)
            rows.append({
                "dataset": ds_name, "feature": feat_name, "fold": fold,
                "group": gname, "n": n,
                "minority_rate": mr,
                "precision_lr": p_lr, "precision_msh3": p_h3,
                "ece_msh3": ece_bins(prob_h3[gidx], yte[gidx]),
                "brier_msh3": brier_score_loss(yte[gidx], prob_h3[gidx]),
                "coverage_all_layers": all_hit[gidx].mean(),
                "coverage_any_layer": any_hit[gidx].mean(),
            })
        print(f"  [fold {fold} done] {ds_name}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("group").agg(
        n=("n", "mean"),
        minority_rate=("minority_rate", "mean"),
        precision_lr=("precision_lr", "mean"),
        precision_msh3=("precision_msh3", "mean"),
        ece_msh3=("ece_msh3", "mean"),
        brier_msh3=("brier_msh3", "mean"),
        coverage_all_layers=("coverage_all_layers", "mean"),
        coverage_any_layer=("coverage_any_layer", "mean"),
    ).reindex(["low", "mid", "high"]).reset_index()
    agg.insert(0, "dataset", ds_name)
    agg.insert(1, "feature", feat_name)
    print(f"[{ds_name} done] elapsed={time.time()-t0:.0f}s", flush=True)
    return agg, df


def main():
    t_all = time.time()
    aggs, folds = [], []
    for ds in AUDIT_FEATURES:
        a, f = run_dataset(ds)
        aggs.append(a)
        folds.append(f)
    out = pd.concat(aggs, ignore_index=True)
    fold_df = pd.concat(folds, ignore_index=True)
    out.to_csv(RESULT_CSV, index=False)
    fold_df.to_csv(RESULT_CSV.replace(".csv", "_foldlevel.csv"), index=False)
    print(out.round(4).to_string(index=False), flush=True)
    print(f"\n[Saved] {RESULT_CSV} | total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
