# -*- coding: utf-8 -*-
"""
Temporal (rolling-window) comparison of PCBC-1 and PCBC (Table 10 of the manuscript)
=====================================================================================

Two arms under the same rolling-window benchmark on the Polish Bankruptcy collection,
five seeds each (seeds 0..4, five-seed mean):

  PCBC-1 : single subspace (K = 1, parallel mode) - the single-layer geometric
           correction without the serial chain
  PCBC   : hierarchical serial re-centering (K = 3, mode='hier') - every layer
           re-selects the pair for the residual left by the previous layer

Tasks (identical to the Temporal Robustness subsection of the manuscript):
  Task 1: train Y1       -> test Y2
  Task 2: train Y1+Y2    -> test Y3
  Task 3: train Y1+Y2+Y3 -> test Y4
  Task 4: train Y1..Y4   -> test Y5

Protocol:
  - standardization with the training-window mean and standard deviation
  - balanced logistic base learner, decision threshold 0.5
  - candidate pairs screened once per training window and reused by the five seeds
  - minority-class Precision / Recall / F1, AUC and PR-AUC

Input files (datasets/): polish_Bankruptcy1.csv .. polish_Bankruptcy5.csv
Dependencies: numpy / pandas / scikit-learn
Run: python code/pcbc_temporal_msh3.py          (all four tasks)
     python code/pcbc_temporal_msh3.py 1 2      (selected tasks only)
Output: pcbc_temporal_msh3_results.csv (one row per task and model, mean +/- std)
"""
import os
import sys
import time
import warnings
import importlib.util

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_temporal_msh3_results.csv")

_spec1 = importlib.util.spec_from_file_location(
    "exp1", os.path.join(DATA_DIR, "code", "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec1)
_spec1.loader.exec_module(exp1)
_spec4 = importlib.util.spec_from_file_location(
    "exp4", os.path.join(DATA_DIR, "code", "pcbc_multisubspace_experiment.py"))
exp4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(exp4)

N_REPEAT = 5          # five seeds (0..4), the five-seed mean reported in the manuscript
TOP_K = 30            # candidate pairs retained by the sparse screen


def eval_temporal_task(train_ds_list, test_ds_name, task_id):
    Xs, ys = [], []
    for ds in train_ds_list:
        X, y, _, _ = exp1.load_dataset(ds)
        Xs.append(X)
        ys.append(y)
    X_train = np.vstack(Xs)
    y_train = np.hstack(ys)
    X_test, y_test, minority_label, test_min_ratio = exp1.load_dataset(test_ds_name)

    mu = X_train.mean(0)
    sigma = X_train.std(0) + 1e-6
    X_tr_n = (X_train - mu) / sigma
    X_te_n = (X_test - mu) / sigma

    # candidate-pair cache: screen once per training window, reused by the five seeds
    lr0 = LogisticRegression(max_iter=500, class_weight="balanced").fit(X_tr_n, y_train)
    pairs5, _ = exp4.select_top_k_pairs(X_tr_n, y_train, lr0.coef_[0],
                                        top_k=TOP_K, K=5, seed=42)

    specs = {"PCBC-1": dict(K=1, mode="parallel"),
             "PCBC":   dict(K=3, mode="hier")}
    records = []
    for seed in range(N_REPEAT):
        for vname, kw in specs.items():
            pairs = pairs5[:kw["K"]] if kw["mode"] == "parallel" else None
            m = exp4.PCBC_MS(seed=seed, **kw)
            m.fit(X_tr_n, y_train, pairs=pairs)
            prob = m.predict_proba(X_te_n)[:, 1]
            records.append((task_id, vname, seed, prob))

    rows = []
    for tid, vname, seed, prob in records:
        y_pred = (prob >= 0.5).astype(int)
        prf = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
        uniq = np.unique(y_test)
        idx_min = 0 if uniq[0] == minority_label else 1
        rows.append({
            "task": tid, "model": vname, "seed": seed,
            "precision": prf[0][idx_min], "recall": prf[1][idx_min], "f1": prf[2][idx_min],
            "auc": roc_auc_score(y_test, prob),
            "auc_pr": average_precision_score(y_test, prob),
        })
    df = pd.DataFrame(rows)
    summ = df.groupby("model").agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
    ).reset_index()
    summ.insert(0, "task", str(task_id))
    summ["test_set"] = test_ds_name
    summ["test_minor_ratio"] = test_min_ratio
    return summ


def main():
    tasks = [
        (["polish_Bankruptcy1"], "polish_Bankruptcy2", 1),
        (["polish_Bankruptcy1", "polish_Bankruptcy2"], "polish_Bankruptcy3", 2),
        (["polish_Bankruptcy1", "polish_Bankruptcy2", "polish_Bankruptcy3"],
         "polish_Bankruptcy4", 3),
        (["polish_Bankruptcy1", "polish_Bankruptcy2", "polish_Bankruptcy3",
          "polish_Bankruptcy4"], "polish_Bankruptcy5", 4),
    ]
    only = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if only:
        tasks = [t for t in tasks if t[2] in only]

    t0 = time.time()
    parts = []
    for train_list, test_name, tid in tasks:
        print(f"===== Task{tid}: train {len(train_list)} year(s) -> test {test_name} =====",
              flush=True)
        summ = eval_temporal_task(train_list, test_name, tid)
        parts.append(summ)
        print(summ.round(4).to_string(index=False), flush=True)

    new = pd.concat(parts, ignore_index=True)
    if os.path.exists(RESULT_CSV) and only:
        old = pd.read_csv(RESULT_CSV)
        old = old[~old["task"].astype(str).isin([str(t[2]) for t in tasks])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(RESULT_CSV, index=False)
    print(f"\n[written] {RESULT_CSV}  |  total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
