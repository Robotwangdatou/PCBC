# Litho-LSIM 2.0 README
## Project Overview
This repository contains the full open-source implementation of Lithography Mask Local Simplex Interaction (Litho-LSIM), the physics-constrained decision support architecture proposed in our paper submitted to Decision Support Systems.
Litho-LSIM integrates a globally regularized logistic regression baseline with a Delaunay triangulation-based local simplex residual correction module (LSIM). Rooted in semiconductor photolithography confinement principles, this framework targets imbalanced credit risk and corporate bankruptcy binary classification tasks. It embeds native auditable geometric constraints and out-of-the-box visualization pipelines for interpretable pairwise feature risk interactions, fully complying with Basel III and EU AI Act transparency requirements for financial decision-making systems.

## File Structure
```
.
├── Litho-LSIM1.1.py       # Full model training, evaluation & visualization source code
├── lsim_single_exp_metric2.csv  # Auto-generated 5-fold evaluation metrics summary
├── lsim_interp_figure/    # Auto-stored Delaunay triangulation PDF plots per fold
└── README.md              # Project documentation
```

## Dependencies
Install required packages before running:
```bash
pip install numpy pandas scikit-learn scipy matplotlib
```
Version requirements (compatible):
- Python ≥ 3.8
- numpy ≥ 1.21
- pandas ≥ 1.3
- scikit-learn ≥ 1.0
- scipy ≥ 1.7
- matplotlib ≥ 3.4

## Dataset Preparation
### Supported Task
Binary credit risk / corporate bankruptcy prediction (imbalanced classification).
### Data Format Rules
1. Place CSV dataset files directly under the project root folder;
2. CSV format requirements:
   - All columns except the last are standardized raw features (model will auto-standardize during training);
   - The **last column** is binary label (0 = non-default/healthy, 1 = default/bankruptcy);
3. Predefined dataset list in code (`dataset_names` variable):
    - australian_credit
    - bank-full (commented by default)
    - german (commented by default)
    - japanese_credit (commented by default)
    - polish_Bankruptcy1~5 (commented by default)
    - taiwan_bankruptcy (commented by default)
    - taiwan_credit (commented by default)
    - US_Bankruptcy (commented by default)
    - give me some credit (commented by default)
4. To run extra datasets: uncomment target names and prepare corresponding `.csv` files.

## Key Hyperparameters
All global hyperparameters are defined at the top of the script, easy to modify:
| Parameter | Default | Description |
| ---- | ---- | ---- |
| `ALPHA` | 0.6 | Fusion weight for LSIM residual correction term |
| `MINOR_WEIGHT` | 1 | Sample weight assigned to minority class in Delaunay residual fitting |
| `N_FOLD` | 5 | Outer stratified cross-validation folds for evaluation |
| `N_REPEAT` | 1 | Experiment repeat times |
| `top_k=30` | Fixed in function | Top candidate feature pairs screened by LR coefficient product for nested CV selection |
| Inner CV split | 3 | 3-fold stratified nested cross-validation for optimal pair selection |

## Run Command
Execute the main script directly:
```bash
python Litho-LSIM1.1.py
```

## Program Execution Flow
1. Create figure save directory `./lsim_interp_figure/` if not exists;
2. Initialize empty metric CSV file `lsim_single_exp_metric2.csv` (write header if missing);
3. Loop over each dataset in `dataset_names`:
   1. Load CSV, split features X and binary label y, calculate class imbalance ratio;
   2. 5-fold stratified outer cross-validation loop:
      - Train-test split + per-fold feature standardization (using train set mean/std only);
      - Fit `LithoBaseLSIM` model:
        a. Balanced Logistic Regression baseline training;
        b. Nested CV select best pairwise feature interaction term;
        c. Build weighted Delaunay LSIM residual surface fitter;
      - Export interpretable model information (selected feature pair, residuals, LR coefficients, triangulation mesh);
      - Generate & save Delaunay triangulation PDF visualization;
      - Predict test set, compute AUC, Precision, Recall, F1-score for minority class;
   3. Aggregate 5-fold metrics (mean ± std), append to result CSV;
   4. Print quantitative interpretability outputs for all folds: selected feature pairs, global residual fill value, fold-wise LR intercept & feature coefficients;
4. After all datasets finish, print completion hint.

## Output Explanation
### 1. Metrics CSV: `lsim_single_exp_metric2.csv`
Each row corresponds to one dataset, columns:
- `dataset`: Dataset name
- `minor_ratio`: Minority class sample proportion (imbalance degree)
- `auc_mean`, `auc_std`: 5-fold ROC-AUC mean and standard deviation
- `pre_mean`, `pre_std`: Minority class Precision mean ± std
- `rec_mean`, `rec_std`: Minority class Recall mean ± std
- `f1_mean`, `f1_std`: Minority class F1-score mean ± std

### 2. Visualization Output (`./lsim_interp_figure/`)
PDF vector plots per outer fold, naming rule:
`{dataset}_fold{fold_id}_pair_{feat1}_{feat2}.pdf`
Plot content:
- Scatter points: blue = non-default (label=0), red = default (label=1);
- Semi-transparent red polygon patches: Delaunay triangles, opacity proportional to local default ratio;
- Light gray mesh: Delaunay triangulation simplex edges;
- Axis: standardized values of the selected optimal feature pair.

### 3. Console Log Output
During running, the terminal prints:
1. Dataset imbalance statistics;
2. Per-fold minority F1 and selected feature interaction pair;
3. Global interpretable summary across all 5 folds:
   - All selected feature pairs (feature names);
   - Residual fusion weight, minority/majority sample weight;
   - Global residual fill value for out-of-mesh query points;
4. Fold-level Logistic Regression intercept and full feature coefficients.

## Core Module Introduction
### 1. Utility Function
`logit_to_prob`: Convert LR logit score to probability with value clipping to avoid numerical overflow.

### 2. LSIMInteraction Class
Delaunay triangulation residual surface fitter:
- `fit()`: Build triangulation mesh on selected feature pair, compute weighted average residual for each triangle vertex;
- `predict()`: Query residual correction value for new samples via triangle simplex lookup; use global mean residual for out-of-convex-hull samples.

### 3. Pair Selection: `select_best_pair_nested`
- Screen top-K feature pairs by product of absolute LR coefficients;
- Nested 3-fold CV to evaluate AUC gain of each candidate pair;
- Return pair with maximum inner validation AUC as optimal interaction term.

### 4. LithoBaseLSIM Class
Scikit-learn compatible classifier (inherits `BaseEstimator, ClassifierMixin`):
- `fit()`: Train LR baseline, select optimal pair, initialize LSIM residual model;
- `decision_function()`: Fuse LR logit and weighted LSIM residual term;
- `predict_proba()` / `predict()`: Probability output and hard classification;
- `get_interpretable_info()`: Export full model interpretable metadata for logging & plotting.

### 5. Visualization Function: `plot_delaunay_figure`
Render Delaunay triangulation interaction plot for interpretability analysis, save as lossless vector PDF.

### 6. Evaluation & Main Pipeline
`evaluate_dataset`: Encapsulate full cross-validation evaluation pipeline for single dataset;
`main`: Entry function to iterate all datasets and aggregate results.

## Customization Guide
1. Add new dataset: Put CSV file in root folder, add dataset name string to `dataset_names` list;
2. Adjust residual fusion strength: Modify `ALPHA` global variable;
3. Adjust minority sample weight in triangulation fitting: Modify `MINOR_WEIGHT`;
4. Change outer CV folds: Modify `N_FOLD`;
5. Modify number of candidate feature pairs for nested CV: Change `top_k` parameter inside `select_best_pair_nested` call;
6. Modify plot style/color/size: Edit parameters inside `plot_delaunay_figure` function.

## Notes
1. The model uses stratified cross-validation throughout to handle class imbalance;
2. Feature standardization is conducted per fold (only training set statistics used to avoid data leakage);
3. Delaunay triangulation only fits on 2 selected interaction features; out-of-convex-hull samples use global average residual as fallback correction value;
4. All figures are saved as PDF vector graphics for high-quality paper visualization;
5. Zero division risks in precision/recall calculation are suppressed with `zero_division=0`.
