# -*- coding: utf-8 -*-
"""
Runtime measurements (Table 13 of the manuscript)
===================================
7 models x 9 datasets x 5 folds: training seconds + inference ms/sample (same protocol as Tables 2 and 3).
Model configurations aligned row by row:
  LR        LogisticRegression(class_weight='balanced', max_iter=500)
  DT        DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42)
  StrictPLTR n_base_trees=20, max_depth=2, C=0.01, saga L1
  EBM       EBMHandcrafted(n_bins=64, n_rounds=20, lr=0.05)
  XGBoost   n_estimators=50, max_depth=3, scale_pos_weight, n_jobs=-1
  PCBC-1    PCBC_MS(seed=42, K=1, mode='parallel'), TOP_K=effective_topk(n)
  PCBC      PCBC_MS(seed=42, K=3, mode='hier'),    TOP_K=effective_topk(n)
Output: pcbc_runtime_full9.csv (aggregate) + pcbc_runtime_full9_fold.csv (fold level)
"""
import os, time, importlib.util
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
warnings.filterwarnings("ignore")

_spec1 = importlib.util.spec_from_file_location("exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec1); _spec1.loader.exec_module(exp1)
_spec4 = importlib.util.spec_from_file_location("exp4", os.path.join(_CODE_DIR, "pcbc_multisubspace_experiment.py"))
exp4 = importlib.util.module_from_spec(_spec4); _spec4.loader.exec_module(exp4)
# reuse the EBM implementation
_specE = importlib.util.spec_from_file_location("expE", os.path.join(_CODE_DIR, "pcbc_allmodels_full9.py"))
expE = importlib.util.module_from_spec(_specE); _specE.loader.exec_module(expE)
# reuse the StrictPLTR implementation (defined in pcbc_dt_pltr_full9.py)
_specP = importlib.util.spec_from_file_location("expP", os.path.join(_CODE_DIR, "pcbc_dt_pltr_full9.py"))
expP = importlib.util.module_from_spec(_specP); _specP.loader.exec_module(expP)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# three representative datasets (same selection as the middleware experiment of the Boundaries section):
#   australian_credit  - balanced, small (n~690)
#   german             - balanced, medium (n~1000)
#   bank-full          - imbalanced, large (n~45000)
DATASETS = ["australian_credit", "german", "bank-full"]

def effective_topk(n):
    if n >= 100000: return 12
    if n >= 30000:  return 20
    return 30

def build_models(topk):
    exp4.TOP_K = topk
    return {
        "LR": LogisticRegression(class_weight="balanced", max_iter=500),
        "DT": DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=42),
        "StrictPLTR": expP.StrictPLTR(),
        "EBM": expE.EBMHandcrafted(n_bins=64, n_rounds=20, learning_rate=0.05),
        "PCBC-1": exp4.PCBC_MS(seed=42, K=1, mode="parallel"),
        "PCBC": exp4.PCBC_MS(seed=42, K=3, mode="hier"),
    }

def run_dataset(ds_name):
    t0 = time.time()
    X, y, minority_label, min_ratio = exp1.load_dataset(ds_name)
    n = len(y)
    topk = effective_topk(n)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        ytr, yte = y[tr], y[te]
        models = build_models(topk)
        if AVAIL_XGB:
            import xgboost
            pos = int((ytr == 1).sum()); neg = int((ytr == 0).sum())
            models["XGBoost"] = xgboost.XGBClassifier(
                n_estimators=50, max_depth=3, scale_pos_weight=neg/max(pos,1),
                random_state=42, n_jobs=-1, eval_metric="logloss")
        for name, m in models.items():
            t = time.time(); m.fit(Xtr, ytr); tr_s = time.time() - t
            t = time.time(); prob = m.predict_proba(Xte)[:, 1]; inf_s = time.time() - t
            rows.append({
                "dataset": ds_name, "model": name, "fold": fold,
                "train_s": round(tr_s, 4),
                "infer_ms_per_sample": round(inf_s / len(Xte) * 1000.0, 4),
                "n_train": len(Xtr), "n_test": len(Xte),
            })
        print(f"  [fold {fold} done] {ds_name}", flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("model").agg(
        train_s_mean=("train_s", "mean"), train_s_std=("train_s", "std"),
        infer_ms_mean=("infer_ms_per_sample", "mean"),
        infer_ms_std=("infer_ms_per_sample", "std"),
    ).reset_index()
    summ.insert(0, "dataset", ds_name)
    summ["n_samples"] = n
    summ["min_ratio"] = min_ratio
    print(f"[{ds_name} done] n={n} elapsed={time.time()-t0:.0f}s", flush=True)
    return summ, df

def main():
    global AVAIL_XGB
    try:
        import xgboost; AVAIL_XGB = True
    except Exception:
        AVAIL_XGB = False
    print("XGBoost:", "OK" if AVAIL_XGB else "MISSING", flush=True)
    t_all = time.time()
    aggs, folds = [], []
    for ds in DATASETS:
        a, f = run_dataset(ds)
        aggs.append(a); folds.append(f)
    out = pd.concat(aggs, ignore_index=True)
    fold_df = pd.concat(folds, ignore_index=True)
    out.to_csv(os.path.join(DATA_DIR, "pcbc_runtime_full9.csv"), index=False)
    fold_df.to_csv(os.path.join(DATA_DIR, "pcbc_runtime_full9_fold.csv"), index=False)
    print("\n===== 9-dataset means by model (train_s / infer ms per sample) =====", flush=True)
    m = out.groupby("model")[["train_s_mean", "infer_ms_mean"]].mean().round(4)
    order = ["LR", "DT", "StrictPLTR", "EBM", "XGBoost", "PCBC-1", "PCBC"]
    order = [x for x in order if x in m.index]
    print(m.reindex(order).to_string(), flush=True)
    print(f"\n[Saved] pcbc_runtime_full9.csv | total {time.time()-t_all:.1f}s", flush=True)

if __name__ == "__main__":
    main()
