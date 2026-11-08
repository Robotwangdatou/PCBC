# -*- coding: utf-8 -*-
"""
Empirical verification of Lemma 1's sign condition (9 datasets x 5 folds x 3 layers)
====================================================================================
Lemma 1 (Per-Layer Contraction) is conditional: on the active set I_k, if the
corrected logit moves toward the target -- (y - sigmoid(s^{k-1})) * (s^k - s^{k-1}) >= 0
-- then the squared training residual is non-increasing. This script measures how
often that condition actually holds on real data, using the exact serial chain of
PCBC_MS(mode='hier', K=3, alpha=0.6) as trained in pcbc_calibration_diagnostic.py.

Layerwise protocol (mirrors the implementation's training loop):
  layer 0 input logit: s_in = lr_logit                     (resid = y - sigmoid(s_in))
  layer k input logit: s_in = lr_logit + alpha * delta_{k-1} (resid = y - sigmoid(s_in))
  delta_k             : barycentric interpolation of that layer's fitted residual
  active set I_k      : training points inside layer k's Delaunay hull
  sign condition      : resid * delta_k >= 0 for i in I_k

Output: pcbc_sign_condition_full9.csv (per dataset per layer) + group means.
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_sign_condition_full9.csv")

DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]
IMBALANCED = {"bank-full", "give me some credit", "polish_Bankruptcy5",
              "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"}
ALPHA = 0.6


def sigmoid(l):
    return 1.0 / (1.0 + np.exp(-np.clip(l, -50, 50)))


def run_dataset(ds_name):
    t0 = time.time()
    X, y, minority_label, min_ratio = exp1.load_dataset(ds_name)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr = (X[tr] - mu) / sg
        ytr = y[tr]
        m = exp4.PCBC_MS(seed=42, K=3, mode="hier", alpha=ALPHA)
        m.fit(Xtr, ytr)
        lr_logit = m.lr.decision_function(Xtr)
        delta_prev = None
        for k, sim in enumerate(m.layers):
            if k == 0:
                s_in = lr_logit
            else:
                s_in = lr_logit + ALPHA * delta_prev
            r_in = ytr - sigmoid(s_in)          # layer k's fitted residual (prob space)
            delta_k = sim.predict(Xtr)          # barycentric correction of layer k
            tri_idx = sim.tri.find_simplex(Xtr[:, sim.pair])
            active = tri_idx != -1              # active set I_k (inside hull)
            rms_in = float(np.sqrt(np.mean(r_in ** 2)))
            sign_ok = (r_in * delta_k) >= 0.0
            n_active = int(active.sum())
            ok_active = sign_ok[active]
            rows.append({
                "dataset": ds_name, "fold": fold, "layer": k,
                "n_train": len(ytr), "n_active": n_active,
                "active_ratio": float(active.mean()),
                "sign_ok_active": float(ok_active.mean()),
                "sign_ok_all": float(sign_ok.mean()),
                "delta_nonzero_active": float((delta_k[active] != 0.0).mean()),
                "rms_input_resid": rms_in,
                "mean_abs_delta": float(np.abs(delta_k[active]).mean()),
            })
            delta_prev = delta_k
        print(f"  [fold {fold} done] {ds_name}", flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("layer").agg(
        n_active=("n_active", "mean"),
        active_ratio=("active_ratio", "mean"),
        sign_ok_active=("sign_ok_active", "mean"),
        sign_ok_all=("sign_ok_all", "mean"),
        delta_nonzero_active=("delta_nonzero_active", "mean"),
        rms_input_resid=("rms_input_resid", "mean"),
        mean_abs_delta=("mean_abs_delta", "mean"),
    ).reset_index()
    summ.insert(0, "dataset", ds_name)
    print(f"[{ds_name} done] elapsed={time.time()-t0:.0f}s", flush=True)
    return summ, df


def main():
    t_all = time.time()
    aggs, folds = [], []
    for ds in DATASETS:
        a, f = run_dataset(ds)
        aggs.append(a)
        folds.append(f)
    out = pd.concat(aggs, ignore_index=True)
    fold_df = pd.concat(folds, ignore_index=True)
    out.to_csv(RESULT_CSV, index=False)
    fold_df.to_csv(RESULT_CSV.replace(".csv", "_foldlevel.csv"), index=False)
    print(out.round(4).to_string(index=False), flush=True)

    print("\n===== 9-dataset means by layer =====", flush=True)
    agg = out.groupby("layer").agg(
        active_ratio=("active_ratio", "mean"),
        sign_ok_active=("sign_ok_active", "mean"),
        sign_ok_all=("sign_ok_all", "mean"),
        rms_input_resid=("rms_input_resid", "mean"),
        mean_abs_delta=("mean_abs_delta", "mean"),
    ).reset_index()
    print(agg.round(4).to_string(index=False), flush=True)

    print("\n===== by imbalance group =====", flush=True)
    out["group"] = np.where(out["dataset"].isin(IMBALANCED), "Imbalanced", "Balanced")
    g = out.groupby(["group", "layer"]).agg(
        sign_ok_active=("sign_ok_active", "mean"),
        active_ratio=("active_ratio", "mean"),
        rms_input_resid=("rms_input_resid", "mean"),
    ).reset_index()
    print(g.round(4).to_string(index=False), flush=True)
    print(f"\n[Saved] {RESULT_CSV} | total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
