Partition-Constrained Barycentric Classifier (PCBC) Project README
Project Overview
PCBC (Partition-Constrained Barycentric Classifier) is a geometry-driven interpretable classification model designed for high-stakes imbalanced credit assessment tasks. It addresses core pain points in traditional credit risk modeling, including the precision-recall tradeoff, insufficient model interpretability, and lack of regulatory compliance. Built on Delaunay simplex partitioning, PCBC adopts a global-linear/local-geometric dual-layer architecture, implementing residual local calibration through hard partition constraints. It balances prediction accuracy, inference efficiency, and audit traceability, fully complying with regulatory requirements such as Basel III and the EU AI Act.
Core Features & Innovations
- Geometry-Driven Partition Constraint Mechanism: Introduces a hard partition indicator operator to strictly confine residual calibration within disjoint Delaunay simplex zones, suppressing full-space residual diffusion, significantly reducing audit false positives, and retaining the ability to identify minority-class samples.
- Global-Local Dual-Layer Architecture: Integrates a global linear baseline model and a local geometric correction module, separating overall population risk trends from localized nonlinear feature interactions to balance global fitting performance and local precise optimization.
- Nested Cross-Validation Sparse Screening: Screens optimal feature pairs via nested cross-validation, avoiding high computational overhead from exhaustive feature pair enumeration while retaining critical two-way feature interaction information.
- End-to-End Audit Traceability: Each prediction result can be linked to three authentic historical samples, enabling complete risk attribution and decision traceability to meet financial audit compliance requirements.
- Efficient Inference Performance: Supports microsecond-level online inference, adapting to the low-latency demands of high-concurrency credit approval systems, while exhibiting good noise robustness and temporal drift adaptability.
Environment Dependencies
This project is developed based on Python 3.8+, with the following required libraries:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
Install all dependencies via the following command:
pip install numpy pandas scikit-learn scipy matplotlib
Project Structure
PCBC/
├── pcbc_model.py          # Core implementation of PCBC model (training, prediction, interpretable information export)
├── pcbc_evaluation.py     # Model evaluation and experimental workflow code (dataset loading, cross-validation, metric calculation)
├── pcbc_interp_figure/    # Directory for saving interpretability visualization results (Delaunay simplex partition diagrams)
├── pcbc_single_exp_metric2.csv  # Output file for experimental result metrics
└── README.md              # Project documentation
Usage Guide
1. Data Preparation
Place credit datasets (CSV format) in the project root directory, with the following requirements:
- The last column is the label column (0 for non-default, 1 for default), and the remaining columns are feature columns;
- Feature columns should be numerical data; no pre-standardization is required (the model will perform standardization based on the training set internally);
- Add/modify dataset names in the dataset_names list in the code to specify the datasets for evaluation.
2. Model Training & Evaluation
Run the main program directly to complete model training, cross-validation evaluation, and result output:
python pcbc_model.py
The program will automatically perform the following operations:
- Load specified datasets and conduct 5-fold stratified cross-validation;
- Train the PCBC model and calculate core metrics including precision, recall, F1-score, and AUC;
- Export model interpretable information (feature pair selection, residual weights, regression coefficients, etc.);
- Generate Delaunay simplex partition visualization diagrams and save them to the pcbc_interp_figure/ directory;
- Summarize evaluation results of all datasets into the pcbc_single_exp_metric2.csv file.
3. Customize Model Parameters
Adjust model performance by modifying global parameters in the code; core parameters are as follows:
- ALPHA: Fusion weight of the local geometric residual correction term, default value 0.6;
- MINOR_WEIGHT: Imbalance weight for minority-class (default) samples, default value 1;
- N_FOLD: Number of cross-validation folds, default value 5;
- top_k: Number of candidate feature pairs screened in nested cross-validation, default value 30.
Experimental Results
Experimental results on nine public retail and corporate credit datasets show that:
- PCBC achieves statistically significantly higher Precision than all baseline models (Logistic Regression, Decision Tree, EBM, XGBoost, etc.);
- Its F1-score is statistically indistinguishable from EBM and XGBoost, and significantly outperforms traditional interpretable models;
- Ablation studies, noise resistance experiments, and temporal drift experiments confirm the rationality of the Delaunay simplex design and model stability.
Application Scenarios
- Credit scoring and default risk prediction for financial institutions;
- Corporate bankruptcy prediction and risk auditing;
- High-stakes imbalanced classification tasks requiring a balance of prediction accuracy, interpretability, and regulatory compliance.
Authors & Acknowledgments
This project is developed by the research team, aiming to provide a theoretically grounded and practical interpretable modeling paradigm for the financial risk control field. If you have any questions or suggestions during usage, please contact the authors for feedback.
License
This project adopts an open-source license, applicable for academic research and commercial use. Please cite the project source when using it.
