# -*- coding: utf-8 -*-
"""
Decision Tree (DT) and Piecewise-Linear Tree Regression (PLTR) baselines on 9 datasets
=====================================================================================
Protocol identical to pcbc_allmodels_full9.py (nine datasets, StratifiedKFold(5, seed=42),
training-fold z-scoring, threshold 0.5, minority-class P/R/F1/AUC/PR-AUC).

Model definitions (interpretable-family baselines):
  DT  : sklearn DecisionTreeClassifier, class_weight='balanced', max_depth=5,
        random_state=42 (depth-limited interpretable tree).
  PLTR: hand-written piecewise-linear tree regression - recursive binary splitting (each node greedily
        picks the feature and split point by weighted SSE, depth 3, minimum 50 samples per leaf), with a
        linear fit inside the leaf (balanced sample weights, logit output, sigmoid probability, 0.5 threshold). Pure numpy/scipy, no external dependency, reproducible.

Output: pcbc_dt_pltr_full9.csv (mean +/- std per dataset and model)
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import numpy as np
import pandas as pd
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
import importlib.util

warnings.filterwarnings("ignore")

_spec1 = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec1)
_spec1.loader.exec_module(exp1)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(DATA_DIR, "pcbc_dt_strictpltr_full9.csv")

DATASETS = ["australian_credit", "bank-full", "german",
            "give me some credit", "japanese_credit", "polish_Bankruptcy5",
            "taiwan_bankruptcy", "taiwan_credit", "US_Bankruptcy"]


def sigmoid(l):
    return 1.0 / (1.0 + np.exp(-np.clip(l, -50, 50)))


class StrictPLTR(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_base_trees: int = 20,
        max_tree_depth: int = 2,
        reg_strength: float = 0.01,
        random_state: int = 42
    ):
        self.n_base_trees = n_base_trees
        self.max_tree_depth = max_tree_depth
        self.reg_strength = reg_strength
        self.random_state = random_state
        self.trees_ = []
        self.tree_leaf_nums = []
        self.fusion_model_ = None

    def _leaf_indicator_matrix(self, tree, X, fixed_leaf_num=None):
        leaf_ids = tree.apply(X)
        _, local_idx = np.unique(leaf_ids, return_inverse=True)
        n_current = np.max(local_idx) + 1
        if fixed_leaf_num is not None:
            mat = np.zeros((X.shape[0], fixed_leaf_num), dtype=np.float64)
            mat[:, :n_current][np.arange(X.shape[0]), local_idx] = 1.0
        else:
            mat = np.zeros((X.shape[0], n_current), dtype=np.float64)
            mat[np.arange(X.shape[0]), local_idx] = 1.0
        return mat

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.trees_ = []
        self.tree_leaf_nums.clear()
        rule_matrices_train = []
        for i in range(self.n_base_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_tree_depth,
                class_weight="balanced",
                random_state=self.random_state
            )
            tree.fit(X, y)
            self.trees_.append(tree)
            leaf_mat = self._leaf_indicator_matrix(tree, X)
            self.tree_leaf_nums.append(leaf_mat.shape[1])
            rule_matrices_train.append(leaf_mat)
        rule_feature_train = np.hstack(rule_matrices_train)
        X_aug_train = np.hstack([X, rule_feature_train])
        self.fusion_model_ = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.reg_strength,
            class_weight="balanced",
            max_iter=5000,
            random_state=self.random_state
        )
        self.fusion_model_.fit(X_aug_train, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        rule_mat_test = []
        for tree, fix_n in zip(self.trees_, self.tree_leaf_nums):
            leaf_mat = self._leaf_indicator_matrix(tree, X, fixed_leaf_num=fix_n)
            rule_mat_test.append(leaf_mat)
        rule_feature_test = np.hstack(rule_mat_test)
        X_aug_test = np.hstack([X, rule_feature_test])
        return self.fusion_model_.predict_proba(X_aug_test)


def run_dataset(ds_name):
    t0 = time.time()
    X, y, minority_label, min_ratio = exp1.load_dataset(ds_name)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg
        ytr, yte = y[tr], y[te]
        models = {
            "DT": DecisionTreeClassifier(class_weight="balanced", max_depth=5,
                                         random_state=42),
            "StrictPLTR": StrictPLTR(),
        }
        for name, m in models.items():
            m.fit(Xtr, ytr)
            prob = m.predict_proba(Xte)[:, 1]
            p, r, f1, _ = precision_recall_fscore_support(
                yte, (prob >= 0.5).astype(int), average="binary", pos_label=1)
            rows.append({
                "dataset": ds_name, "model": name, "fold": fold,
                "precision": p, "recall": r, "f1": f1,
                "auc": roc_auc_score(yte, prob),
                "auc_pr": average_precision_score(yte, prob),
            })
        print(f"  [fold {fold} done] {ds_name}", flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("model").agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
    ).reset_index()
    summ.insert(0, "dataset", ds_name)
    summ.insert(1, "n_samples", len(y))
    summ["test_minor_ratio"] = min_ratio
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
    print("\n===== 9-dataset means by model =====", flush=True)
    means = out.groupby("model")[["precision_mean", "recall_mean", "f1_mean",
                                  "auc_mean", "auc_pr_mean"]].mean().round(4)
    print(means.to_string(), flush=True)
    print(f"\n[Saved] {RESULT_CSV} | total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
