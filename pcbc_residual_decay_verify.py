# -*- coding: utf-8 -*-
"""
Residual-Decay Verification for Lemma 1 & Proposition (serial vs parallel)
==========================================================================
Protocol identical to pcbc_msh3_full9.py (nine datasets, 5-fold, seed=42, top_k 12/20/30,
ALPHA=0.6、use_bary=True）。
Recorded:
  serial PCBC (hier, K=3): training/test residual-norm sequence ||r0||,||r1||,||r2||,||r3||
  parallel MS3 (K=3): final training/test residual norm ||r_par|| (compared with ||r0||)
Output: pcbc_residual_decay_verify.csv (norms per dataset and fold, with the contraction rate).
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

_spec = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)
_spec4 = importlib.util.spec_from_file_location(
    "exp4", os.path.join(_CODE_DIR, "pcbc_multisubspace_experiment.py"))
exp4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(exp4)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_residual_decay_verify.csv")


DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]

ALPHA = exp4.ALPHA


def effective_topk(n):
    if n >= 100000:
        return 12
    if n >= 30000:
        return 20
    return 30


def l2norm(a):
    return float(np.linalg.norm(a))


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

        lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        lr.fit(Xtr, ytr)
        z0_tr = lr.decision_function(Xtr)
        p0_tr = exp1.logit_to_prob(z0_tr)
        r0_tr = ytr - p0_tr
        norm0_tr = l2norm(r0_tr)
        z0_te = lr.decision_function(Xte)
        p0_te = exp1.logit_to_prob(z0_te)
        norm0_te = l2norm(yte - p0_te)

        # ---------- serial PCBC (hier, K=3): residual-norm sequence layer by layer ----------
        prob_tr = p0_tr.copy()
        z_te = z0_te.copy()
        norms_tr = [norm0_tr]
        norms_te = [norm0_te]
        for k in range(3):
            resid_k = ytr - prob_tr
            pair_k, _ = exp4.select_top_k_pairs(
                Xtr, ytr, lr.coef_[0], top_k=topk, K=1, seed=42,
                target_resid=resid_k)
            sim = exp1.TemporalLSIM(pair_k[0], use_bary=True, seed=42)
            sim.fit(Xtr, resid_k, ytr, np.ones(len(ytr)))
            # training update (hier protocol identical to PCBC_MS.fit)
            z_tr = lr.decision_function(Xtr) + ALPHA * sim.predict(Xtr)
            prob_tr = exp1.logit_to_prob(z_tr)
            norms_tr.append(l2norm(ytr - prob_tr))
            # accumulate on the test side
            z_te = z_te + ALPHA * sim.predict(Xte)
            norms_te.append(l2norm(yte - exp1.logit_to_prob(z_te)))

        # ---------- parallel MS3 (K=3): final residual norm ----------
        pairs3, _ = exp4.select_top_k_pairs(
            Xtr, ytr, lr.coef_[0], top_k=topk, K=3, seed=42, target_resid=None)
        inter = exp4.MultiLSIM(pairs3, use_bary=True, seed=42)
        inter.fit(Xtr, r0_tr, ytr, np.ones(len(ytr)))
        z_par_tr = lr.decision_function(Xtr) + ALPHA * inter.predict(Xtr)
        z_par_te = lr.decision_function(Xte) + ALPHA * inter.predict(Xte)
        norm_par_tr = l2norm(ytr - exp1.logit_to_prob(z_par_tr))
        norm_par_te = l2norm(yte - exp1.logit_to_prob(z_par_te))

        rows.append({
            "dataset": ds_name, "fold": fold,
            "serial_norm_train_0": norms_tr[0], "serial_norm_train_1": norms_tr[1],
            "serial_norm_train_2": norms_tr[2], "serial_norm_train_3": norms_tr[3],
            "serial_norm_test_0": norms_te[0], "serial_norm_test_1": norms_te[1],
            "serial_norm_test_2": norms_te[2], "serial_norm_test_3": norms_te[3],
            "par_norm_train": norm_par_tr, "par_norm_test": norm_par_te,
            "serial_train_decay": norms_tr[3] / norms_tr[0],
            "par_train_decay": norm_par_tr / norm0_tr,
        })
        print(f"  [fold {fold} done] {ds_name}", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("dataset").agg(
        sn0=("serial_norm_train_0", "mean"), sn1=("serial_norm_train_1", "mean"),
        sn2=("serial_norm_train_2", "mean"), sn3=("serial_norm_train_3", "mean"),
        st0=("serial_norm_test_0", "mean"), st3=("serial_norm_test_3", "mean"),
        pn=("par_norm_train", "mean"), pnt=("par_norm_test", "mean"),
        s_decay=("serial_train_decay", "mean"), p_decay=("par_train_decay", "mean"),
    ).reset_index()
    agg.insert(1, "n_samples", n)
    agg.insert(2, "test_minor_ratio", min_ratio)
    agg["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[{ds_name} done] n={n} elapsed={agg['elapsed_s'].iloc[0]}s", flush=True)
    return agg, df


def main():
    t_all = time.time()
    aggs, dfs = [], []
    for ds in DATASETS:
        a, d = run_dataset(ds)
        aggs.append(a)
        dfs.append(d)
    out = pd.concat(aggs, ignore_index=True)
    out.to_csv(RESULT_CSV, index=False)
    detail = pd.concat(dfs, ignore_index=True)
    detail.to_csv(RESULT_CSV.replace(".csv", "_detail.csv"), index=False)

    print("\n===== per-dataset serial train-norm sequence (mean over folds) =====", flush=True)
    print(out.round(4).to_string(index=False), flush=True)
    print("\n===== 9-dataset means =====", flush=True)
    mean_row = out[["sn0", "sn1", "sn2", "sn3", "st0", "st3",
                    "pn", "pnt", "s_decay", "p_decay"]].mean()
    print(mean_row.round(4).to_string(), flush=True)
    print(f"\n[Saved] {RESULT_CSV} (+_detail.csv) | total {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
