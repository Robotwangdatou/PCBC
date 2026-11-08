# -*- coding: utf-8 -*-
"""Supplementary run: PCBC-MS-H3 (hierarchical serial) on the static Y5 setting; results are appended to the direction-4 main CSV."""
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
import importlib.util
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore")
_spec = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)
_spec4 = importlib.util.spec_from_file_location(
    "exp4", os.path.join(_CODE_DIR, "pcbc_multisubspace_experiment.py"))
exp4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(exp4)

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pcbc_multisubspace_results.csv")

X, y, minority_label, min_ratio = exp1.load_dataset("polish_Bankruptcy5")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
records = []
for fold, (tr, te) in enumerate(skf.split(X, y)):
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
    m = exp4.PCBC_MS(seed=42, K=3, mode="hier")
    m.fit(Xtr, y[tr])
    prob = m.predict_proba(Xte)[:, 1]
    y_pred = (prob >= 0.5).astype(int)
    prf = precision_recall_fscore_support(y[te], y_pred, average=None, zero_division=0)
    uniq = np.unique(y[te]); idx_min = 0 if uniq[0] == minority_label else 1
    records.append({
        "fold": fold, "model": "PCBC-MS-H3",
        "precision": prf[0][idx_min], "recall": prf[1][idx_min], "f1": prf[2][idx_min],
        "auc": roc_auc_score(y[te], prob), "auc_pr": average_precision_score(y[te], prob),
    })
df = pd.DataFrame(records)
summ = df.groupby("model").agg(
    precision_mean=("precision", "mean"), precision_std=("precision", "std"),
    f1_mean=("f1", "mean"), f1_std=("f1", "std"),
    auc_mean=("auc", "mean"), auc_std=("auc", "std"),
    auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
).reset_index()
summ.insert(0, "task", "Y5-static")
summ["test_set"] = "polish_Bankruptcy5"
summ["test_minor_ratio"] = min_ratio
old = pd.read_csv(CSV)
out = pd.concat([old, summ], ignore_index=True)
out.to_csv(CSV, index=False)
print(summ.round(4).to_string(index=False))
print("[appended]", CSV)
