# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 07:54:40 2026
Litho-LSIM2.0
@author: robot
"""
import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy.spatial import Delaunay
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

FIG_SAVE_DIR = "./lsim_interp_figure"
if not os.path.exists(FIG_SAVE_DIR):
    os.makedirs(FIG_SAVE_DIR)

# ======================= Utility Functions =======================
def logit_to_prob(l):
    return 1.0 / (1.0 + np.exp(-np.clip(l, -50, 50)))

# ======================= Delaunay LSIM Interaction Module =======================
class LSIMInteraction:
    def __init__(self, pair, weight_major=1.0, weight_minor=1.0):
        self.pair = pair
        self.w_maj = weight_major
        self.w_min = weight_minor
        self.tri = None
        self.xy_train = None
        self.resid_train = None
        self.w_train = None
        self.fill = 0.0

    def fit(self, X, resid, y):
        xy = X[:, self.pair].copy()
        eps = 1e-6
        xy += np.random.randn(*xy.shape)
        self.fill = np.mean(resid)

        uniq, cnt = np.unique(y, return_counts=True)
        minority_label = uniq[np.argmin(cnt)]
        w = np.where(y == minority_label, self.w_min, self.w_maj)

        self.tri = Delaunay(xy)
        self.xy_train = xy
        self.resid_train = resid
        self.w_train = w

    def predict(self, X):
        xy_query = X[:, self.pair]
        tri_idx = self.tri.find_simplex(xy_query)
        pred = np.full(len(xy_query), self.fill)
        valid_mask = tri_idx != -1

        for i in np.nonzero(valid_mask)[0]:
            pts = self.tri.simplices[tri_idx[i]]
            r_vals = self.resid_train[pts]
            w_vals = self.w_train[pts]
            pred[i] = np.sum(r_vals * w_vals) / np.sum(w_vals)
        return pred


# ======================= Nested CV Feature Pair Selection =======================
def select_best_pair_nested(X_tr, y_tr, lr_coef, top_k=30):
    D = X_tr.shape[1]
    coef = np.abs(lr_coef)
    cand = {}
    for i in range(D):
        for j in range(i + 1, D):
            cand[(i, j)] = coef[i] * coef[j]
    top = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:top_k]
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pair_scores = {}
    for (i, j), _ in top:
        aucs = []
        for itr_inner, ival_inner in inner_cv.split(X_tr, y_tr):
            X_inner_tr = X_tr[itr_inner]
            y_inner_tr = y_tr[itr_inner]
            X_inner_val = X_tr[ival_inner]
            y_inner_val = y_tr[ival_inner]
            
            lr = LogisticRegression(max_iter=500)
            lr.fit(X_inner_tr, y_inner_tr)
            resid = y_inner_tr - lr.predict_proba(X_inner_tr)[:, 1]
            lsim = LSIMInteraction((i, j))
            lsim.fit(X_inner_tr, resid, y_inner_tr)
            
            logit = lr.decision_function(X_inner_val) + 0.5 * lsim.predict(X_inner_val)
            prob = logit_to_prob(logit)
            try:
                aucs.append(roc_auc_score(y_inner_val, prob))
            except:
                aucs.append(0.5)
        pair_scores[(i, j)] = np.mean(aucs)
    return max(pair_scores, key=pair_scores.get)

# ======================= Litho-Base LSIM Classifier =======================
class LithoBaseLSIM(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.6, seed=42, weight_minor=1.0):
        self.alpha = alpha
        self.seed = seed
        self.weight_minor = weight_minor
        self.lr = None
        self.pair = None
        self.inter = None

    def fit(self, X, y):
        self.lr = LogisticRegression(max_iter=500, class_weight="balanced")
        self.lr.fit(X, y)
        self.pair = select_best_pair_nested(X, y, self.lr.coef_[0], top_k=30)
        prob = self.lr.predict_proba(X)[:, 1]
        resid = y - prob
        self.inter = LSIMInteraction(
            pair=self.pair,
            weight_minor=self.weight_minor
        )
        self.inter.fit(X, resid, y)
        return self

    def decision_function(self, X):
        logit = self.lr.decision_function(X)
        logit += self.alpha * self.inter.predict(X)
        return logit

    def predict_proba(self, X):
        logit = self.decision_function(X)
        p = logit_to_prob(logit)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_interpretable_info(self):
        info = {
            "selected_feature_pair": self.pair,
            "residual_alpha_weight": self.alpha,
            "minority_sample_weight": self.inter.w_min,
            "majority_sample_weight": self.inter.w_maj,
            "global_residual_fill_value": float(self.inter.fill),
            "lr_coef": self.lr.coef_[0].tolist(),
            "intercept": float(self.lr.intercept_[0]),
            "xy_train": self.inter.xy_train.copy(),
            "tri": self.inter.tri
        }
        return info


# ======================= Visualization: Delaunay Triangulation Plot =======================
def plot_delaunay_figure(ds_name, fold_id, info, y_train, feat_name_map):
    xy = info["xy_train"]
    tri = info["tri"]
    pair_idx = info["selected_feature_pair"]
    f0_name = feat_name_map[pair_idx[0]]
    f1_name = feat_name_map[pair_idx[1]]

    fig, ax = plt.subplots(figsize=(7, 6))
    mask_neg = y_train == 0
    mask_pos = y_train == 1
    ax.scatter(xy[mask_neg, 0], xy[mask_neg, 1], c="#1f77b4", s=15, alpha=0.7, label="Non-default")
    ax.scatter(xy[mask_pos, 0], xy[mask_pos, 1], c="#d62728", s=15, alpha=0.8, label="Default")

    for simplex in tri.simplices:
        vertex_coords = xy[simplex]
        vertex_labels = y_train[simplex]
        default_ratio = sum(vertex_labels) / len(vertex_labels)
        fill_alpha = default_ratio * 0.22
        poly = Polygon(vertex_coords, facecolor="#ff4444", edgecolor="none", alpha=fill_alpha)
        ax.add_patch(poly)

    ax.triplot(xy[:, 0], xy[:, 1], tri.simplices, lw=0.3, c="#555555", alpha=0.4)

    ax.set_xlabel(f"{f0_name} (standardized)")
    ax.set_ylabel(f"{f1_name} (standardized)")
    ax.set_title(f"{ds_name} Fold {fold_id} | Delaunay triangulation, feature pair: ({f0_name}, {f1_name})")
    ax.legend(loc="best")
    plt.tight_layout()

    save_path = os.path.join(FIG_SAVE_DIR, f"{ds_name}_fold{fold_id}_pair_{f0_name}_{f1_name}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[Visualization Saved] {save_path}")

# ======================= Global Experiment Configurations =======================
dataset_names = [
    "australian_credit",
    #"bank-full",
    #"german",
    #"japanese_credit",
    #"polish_Bankruptcy1",
    #"polish_Bankruptcy2",
    #"polish_Bankruptcy3",
    #"polish_Bankruptcy4",
    #"polish_Bankruptcy5",
    #"taiwan_bankruptcy",
    #"taiwan_credit",
    #"US_Bankruptcy",
    #"give me some credit"
]
ALPHA = 0.6
MINOR_WEIGHT = 1
N_REPEAT = 1
N_FOLD = 5
RESULT_CSV = "lsim_single_exp_metric2.csv"

# ======================= Single Dataset Evaluation Pipeline =======================
def evaluate_dataset(ds_name):
    file_path = f"{ds_name}.csv"
    if not os.path.exists(file_path):
        print(f"[Warning] File missing: {file_path}, skip dataset")
        return None

    print(f"\n==================== Dataset: {ds_name} ====================")
    df = pd.read_csv(file_path)
    feat_cols = df.columns[:-1].tolist()
    feat_name_map = {idx: name for idx, name in enumerate(feat_cols)}
    label_col = df.columns[-1]
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col]).astype(np.float32).values

    uniq, cnt = np.unique(y, return_counts=True)
    minority_label = uniq[np.argmin(cnt)]
    min_ratio = cnt.min() / len(y)
    print(f"Minority label = {minority_label}, imbalance ratio = {min_ratio:.2%}")

    auc_list, pre_list, rec_list, f1_list = [], [], [], []
    interp_record = []

    seed = 42
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0) + 1e-6
        X_tr = (X_tr - mu) / sigma
        X_te = (X_te - mu) / sigma

        model = LithoBaseLSIM(alpha=ALPHA, seed=seed, weight_minor=MINOR_WEIGHT)
        model.fit(X_tr, y_tr)
        interp = model.get_interpretable_info()
        interp_record.append(interp)

        plot_delaunay_figure(ds_name, fold_idx+1, interp, y_tr, feat_name_map)

        prob = model.predict_proba(X_te)[:, 1]
        y_pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y_te, prob)
        prec_rec_f1 = precision_recall_fscore_support(y_te, y_pred, average=None, zero_division=0)
        idx_min = 0 if uniq[0]==minority_label else 1
        pre, rec, f1 = prec_rec_f1[0][idx_min], prec_rec_f1[1][idx_min], prec_rec_f1[2][idx_min]

        auc_list.append(auc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)
        p_idx = interp['selected_feature_pair']
        p_name = (feat_name_map[p_idx[0]], feat_name_map[p_idx[1]])
        print(f"Fold{fold_idx+1} | F1={f1:.4f}, selected feature pair = {p_name}")

    def mean_std(arr):
        return round(np.mean(arr),4), round(np.std(arr),4)
    auc_mean, auc_std = mean_std(auc_list)
    pre_mean, pre_std = mean_std(pre_list)
    rec_mean, rec_std = mean_std(rec_list)
    f1_mean, f1_std = mean_std(f1_list)

    row_data = {
        "dataset": ds_name,
        "minor_ratio": round(min_ratio,4),
        "auc_mean":auc_mean,"auc_std":auc_std,
        "pre_mean":pre_mean,"pre_std":pre_std,
        "rec_mean":rec_mean,"rec_std":rec_std,
        "f1_mean":f1_mean,"f1_std":f1_std
    }

    print("\n========== Quantitative Interpretable Output (5-Fold Summary) ==========")
    all_pairs_idx = [item["selected_feature_pair"] for item in interp_record]
    all_pairs_name = []
    for (i,j) in all_pairs_idx:
        all_pairs_name.append((feat_name_map[i], feat_name_map[j]))
    fill_vals = [round(item["global_residual_fill_value"],4) for item in interp_record]

    print(f"Selected feature pairs (feature names): {all_pairs_name}")
    print(f"Residual fusion alpha = {ALPHA}")
    print(f"Minority weight = {MINOR_WEIGHT}, Majority weight = {interp_record[0]['majority_sample_weight']}")
    print(f"Global residual fill per fold: {fill_vals}")

    print("\n--- Fold-wise Logistic Regression Coefficients ---")
    for fold_num, info in enumerate(interp_record, start=1):
        coefs = info["lr_coef"]
        intercept = round(info["intercept"], 4)
        feat_coef_pairs = [(feat_name_map[i], round(coefs[i], 4)) for i in range(len(coefs))]
        print(f"Fold {fold_num} intercept = {intercept}")
        print(f"Fold {fold_num} feature coefficients: {feat_coef_pairs}")

    print("=========================================================================\n")
    print(f"Dataset {ds_name} finished, average F1 = {f1_mean:.4f}")
    return row_data

# ======================= Main Entry Function =======================
def main():
    if not os.path.exists(RESULT_CSV):
        header = [
            "dataset","minor_ratio",
            "auc_mean","auc_std",
            "pre_mean","pre_std",
            "rec_mean","rec_std",
            "f1_mean","f1_std"
        ]
        pd.DataFrame(columns=header).to_csv(RESULT_CSV, index=False)

    for ds in dataset_names:
        res = evaluate_dataset(ds)
        if res is None:
            continue
        pd.DataFrame([res]).to_csv(RESULT_CSV, mode="a", header=False, index=False)
        print(f"Metrics saved into {RESULT_CSV}")
    print(f"\nAll datasets finished. All 5-fold triangulation figures stored at {FIG_SAVE_DIR}")

if __name__ == "__main__":
    main()