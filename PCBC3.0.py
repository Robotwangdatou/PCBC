# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 07:54:40 2026
PCBC Model Implementation (Partition-Constrained Barycentric Classifier)
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

FIG_SAVE_DIR = "./pcbc_interp_figure"
if not os.path.exists(FIG_SAVE_DIR):
    os.makedirs(FIG_SAVE_DIR)

# ======================= Common Utility Functions =======================
def logit_to_prob(l):
    """Convert logit score to probability with numerical clipping"""
    return 1.0 / (1.0 + np.exp(-np.clip(l, -50, 50)))

# ======================= Delaunay Partition Residual Interaction Module (PCBC Core Component) =======================
class LSIMInteraction:
    """
    Local Spatial Interaction Module for PCBC
    Realizes localized barycentric residual calibration within independent Delaunay simplex regions
    """
    def __init__(self, pair, weight_major=1.0, weight_minor=1.0):
        self.pair = pair               # Selected two-dimensional feature interaction pair
        self.w_maj = weight_major      # Weight for majority (non-default) samples
        self.w_min = weight_minor      # Weight for minority (default) samples
        self.tri = None                # Delaunay triangulation object of training feature subspace
        self.xy_train = None           # 2D standardized feature pair training data
        self.resid_train = None        # Global linear model training residuals
        self.w_train = None            # Per-sample imbalance weight array
        self.fill = 0.0                # Global average residual fill value for out-of-bound samples

    def fit(self, X, resid, y):
        """Fit local simplex residual correction module on training set"""
        xy = X[:, self.pair].copy()
        eps = 1e-6
        xy += np.random.randn(*xy.shape)  # Avoid coplanar singular simplices
        self.fill = np.mean(resid)        # Global fallback residual for points outside all simplices

        # Assign imbalance weight based on minority label
        uniq, cnt = np.unique(y, return_counts=True)
        minority_label = uniq[np.argmin(cnt)]
        w = np.where(y == minority_label, self.w_min, self.w_maj)

        # Construct Delaunay triangulation for 2D feature subspace
        self.tri = Delaunay(xy)
        self.xy_train = xy
        self.resid_train = resid
        self.w_train = w

    def predict(self, X):
        """Predict localized residual correction term for input samples"""
        xy_query = X[:, self.pair]
        tri_idx = self.tri.find_simplex(xy_query)
        pred = np.full(len(xy_query), self.fill)  # Initialize with global residual fill
        valid_mask = tri_idx != -1                # Mask samples falling inside triangulation area

        # Barycentric weighted residual aggregation within each simplex
        for i in np.nonzero(valid_mask)[0]:
            pts = self.tri.simplices[tri_idx[i]]
            r_vals = self.resid_train[pts]
            w_vals = self.w_train[pts]
            pred[i] = np.sum(r_vals * w_vals) / np.sum(w_vals)
        return pred


# ======================= Nested Cross-Validation Sparse Feature Pair Screening (PCBC Pipeline Module) =======================
def select_best_pair_nested(X_tr, y_tr, lr_coef, top_k=30):
    """
    Nested cross-validation sparse screening for optimal two-way feature interaction pair
    Step1: Rank candidate pairs by product of linear regression absolute coefficients
    Step2: Inner 3-fold stratified CV to select pair with highest AUC gain
    """
    D = X_tr.shape[1]
    coef = np.abs(lr_coef)
    cand = {}
    # Enumerate all unique feature pairs
    for i in range(D):
        for j in range(i + 1, D):
            cand[(i, j)] = coef[i] * coef[j]
    # Retain top-K candidate pairs
    top = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:top_k]
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pair_scores = {}
    # Inner cross validation evaluation for each candidate pair
    for (i, j), _ in top:
        aucs = []
        for itr_inner, ival_inner in inner_cv.split(X_tr, y_tr):
            X_inner_tr = X_tr[itr_inner]
            y_inner_tr = y_tr[itr_inner]
            X_inner_val = X_tr[ival_inner]
            y_inner_val = y_tr[ival_inner]
            
            # Baseline global linear logistic model
            lr = LogisticRegression(max_iter=500)
            lr.fit(X_inner_tr, y_inner_tr)
            resid = y_inner_tr - lr.predict_proba(X_inner_tr)[:, 1]
            # Initialize local simplex correction module
            lsim = LSIMInteraction((i, j))
            lsim.fit(X_inner_tr, resid, y_inner_tr)
            
            # Fusion of global linear logit and local simplex residual correction
            logit = lr.decision_function(X_inner_val) + 0.5 * lsim.predict(X_inner_val)
            prob = logit_to_prob(logit)
            try:
                aucs.append(roc_auc_score(y_inner_val, prob))
            except:
                aucs.append(0.5)
        pair_scores[(i, j)] = np.mean(aucs)
    # Return feature pair with maximum inner-CV AUC
    return max(pair_scores, key=pair_scores.get)

# ======================= Partition-Constrained Barycentric Classifier (PCBC Main Model) =======================
class LithoBaseLSIM(BaseEstimator, ClassifierMixin):
    """
    PCBC: Partition-Constrained Barycentric Classifier
    Global-linear & local-geometric dual-layer interpretable credit risk model
    Core modules: Global regularized logistic baseline, Delaunay simplex local residual correction
    """
    def __init__(self, alpha=0.6, seed=42, weight_minor=1.0):
        self.alpha = alpha               # Fusion weight for local simplex residual correction term
        self.seed = seed
        self.weight_minor = weight_minor # Imbalance weight for minority default samples
        self.lr = None                   # Global linear logistic regression baseline
        self.pair = None                 # Selected optimal 2D feature interaction pair
        self.inter = None                # Local spatial simplex residual correction module

    def fit(self, X, y):
        """End-to-end training pipeline of PCBC"""
        # Step 1: Train balanced global linear baseline model
        self.lr = LogisticRegression(max_iter=500, class_weight="balanced")
        self.lr.fit(X, y)
        # Step 2: Nested CV screening to select optimal feature pair for geometric subspace fitting
        self.pair = select_best_pair_nested(X, y, self.lr.coef_[0], top_k=30)
        # Step3: Calculate residual between true label and linear model predicted probability
        prob = self.lr.predict_proba(X)[:, 1]
        resid = y - prob
        # Step4: Train Delaunay partition local barycentric residual correction module
        self.inter = LSIMInteraction(
            pair=self.pair,
            weight_minor=self.weight_minor
        )
        self.inter.fit(X, resid, y)
        return self

    def decision_function(self, X):
        """Fused logit score: global linear term + weighted local simplex residual correction"""
        logit = self.lr.decision_function(X)
        logit += self.alpha * self.inter.predict(X)
        return logit

    def predict_proba(self, X):
        """Output predicted default probability"""
        logit = self.decision_function(X)
        p = logit_to_prob(logit)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        """Hard classification prediction with 0.5 probability threshold"""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_interpretable_info(self):
        """Export full interpretable structural information of trained PCBC model"""
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


# ======================= Visualization: Delaunay Simplex Partition Diagram for PCBC Interpretability =======================
def plot_delaunay_figure(ds_name, fold_id, info, y_train, feat_name_map):
    """
    Visualize Delaunay triangulation partition of the selected two-dimensional feature subspace
    Color gradient of simplex regions corresponds to default sample proportion, for model interpretability analysis
    """
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

    # Draw simplex regions with transparency proportional to default sample ratio
    for simplex in tri.simplices:
        vertex_coords = xy[simplex]
        vertex_labels = y_train[simplex]
        default_ratio = sum(vertex_labels) / len(vertex_labels)
        fill_alpha = default_ratio * 0.22
        poly = Polygon(vertex_coords, facecolor="#ff4444", edgecolor="none", alpha=fill_alpha)
        ax.add_patch(poly)

    # Draw Delaunay triangulation mesh lines
    ax.triplot(xy[:, 0], xy[:, 1], tri.simplices, lw=0.3, c="#555555", alpha=0.4)

    ax.set_xlabel(f"{f0_name} (standardized)")
    ax.set_ylabel(f"{f1_name} (standardized)")
    ax.set_title(f"{ds_name} Fold {fold_id} | Delaunay simplex partition, feature pair: ({f0_name}, {f1_name})")
    ax.legend(loc="best")
    plt.tight_layout()

    save_path = os.path.join(FIG_SAVE_DIR, f"{ds_name}_fold{fold_id}_pair_{f0_name}_{f1_name}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[Interpretation Figure Saved] {save_path}")

# ======================= Global Experiment Hyperparameter Configurations for PCBC =======================
dataset_names = [
    "australian_credit",
    #"bank-full",
    #"german_credit",
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
ALPHA = 0.6               # Weight coefficient of local geometric residual correction term in PCBC
MINOR_WEIGHT = 1          # Imbalance loss weight for minority default samples
N_REPEAT = 10              # Number of repeated outer cross-validation runs
N_FOLD = 5                # 5-fold stratified outer cross-validation
RESULT_CSV = "pcbc_single_exp_metric2.csv"  # Output path of all evaluation metrics

# ======================= Single Credit Dataset 5-Fold Evaluation Pipeline for PCBC =======================
def evaluate_dataset(ds_name):
    """Full evaluation workflow of PCBC on a single credit risk dataset"""
    file_path = f"{ds_name}.csv"
    if not os.path.exists(file_path):
        print(f"[Warning] Dataset file missing: {file_path}, skip this dataset")
        return None

    print(f"\n==================== Dataset: {ds_name} ====================")
    df = pd.read_csv(file_path)
    feat_cols = df.columns[:-1].tolist()
    feat_name_map = {idx: name for idx, name in enumerate(feat_cols)}
    label_col = df.columns[-1]
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col]).astype(np.float32).values

    # Print dataset imbalance statistics
    uniq, cnt = np.unique(y, return_counts=True)
    minority_label = uniq[np.argmin(cnt)]
    min_ratio = cnt.min() / len(y)
    print(f"Minority default label = {minority_label}, class imbalance ratio = {min_ratio:.2%}")

    # Storage for fold-wise evaluation metrics and model interpretation records
    auc_list, pre_list, rec_list, f1_list = [], [], [], []
    interp_record = []

    seed = 42
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        # Standardization based on training set statistics
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0) + 1e-6
        X_tr = (X_tr - mu) / sigma
        X_te = (X_te - mu) / sigma

        # Initialize and train PCBC model
        model = LithoBaseLSIM(alpha=ALPHA, seed=seed, weight_minor=MINOR_WEIGHT)
        model.fit(X_tr, y_tr)
        interp = model.get_interpretable_info()
        interp_record.append(interp)

        # Save Delaunay partition interpretability figure for current fold
        plot_delaunay_figure(ds_name, fold_idx+1, interp, y_tr, feat_name_map)

        # Test set prediction and metric calculation
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
        print(f"Fold{fold_idx+1} | Minority F1={f1:.4f}, selected feature pair = {p_name}")

    # Calculate mean and standard deviation across 5 folds
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

    print("\n========== PCBC Interpretable Structural Output (5-Fold Summary) ==========")
    all_pairs_idx = [item["selected_feature_pair"] for item in interp_record]
    all_pairs_name = []
    for (i,j) in all_pairs_idx:
        all_pairs_name.append((feat_name_map[i], feat_name_map[j]))
    fill_vals = [round(item["global_residual_fill_value"],4) for item in interp_record]

    print(f"Selected feature interaction pairs across folds (feature names): {all_pairs_name}")
    print(f"Local residual fusion coefficient alpha = {ALPHA}")
    print(f"Minority sample imbalance weight = {MINOR_WEIGHT}, Majority sample weight = {interp_record[0]['majority_sample_weight']}")
    print(f"Global average residual fill value per fold: {fill_vals}")

    print("\n--- Fold-wise Global Linear Logistic Regression Coefficients (PCBC Baseline) ---")
    for fold_num, info in enumerate(interp_record, start=1):
        coefs = info["lr_coef"]
        intercept = round(info["intercept"], 4)
        feat_coef_pairs = [(feat_name_map[i], round(coefs[i], 4)) for i in range(len(coefs))]
        print(f"Fold {fold_num} linear intercept = {intercept}")
        print(f"Fold {fold_num} feature linear coefficients: {feat_coef_pairs}")

    print("=========================================================================\n")
    print(f"Dataset {ds_name} evaluation completed, average minority F1 = {f1_mean:.4f}")
    return row_data

# ======================= Main Experiment Entry Function for PCBC =======================
def main():
    """Main execution function for batch PCBC evaluation on all credit datasets"""
    # Create empty metric CSV file with header if not exists
    if not os.path.exists(RESULT_CSV):
        header = [
            "dataset","minor_ratio",
            "auc_mean","auc_std",
            "pre_mean","pre_std",
            "rec_mean","rec_std",
            "f1_mean","f1_std"
        ]
        pd.DataFrame(columns=header).to_csv(RESULT_CSV, index=False)

    # Evaluate each dataset sequentially and append metrics to CSV
    for ds in dataset_names:
        res = evaluate_dataset(ds)
        if res is None:
            continue
        pd.DataFrame([res]).to_csv(RESULT_CSV, mode="a", header=False, index=False)
        print(f"Dataset metrics saved to {RESULT_CSV}")
    print(f"\nAll datasets evaluation finished. Delaunay simplex partition interpretability figures saved to {FIG_SAVE_DIR}")

if __name__ == "__main__":
    main()