<div align="center">

Credit Card Fraud Detection

Imbalanced classification for identifying rare fraudulent credit-card transactions.

<p>
  <img src="https://img.shields.io/badge/Transactions-284%2C807-0969da?style=flat-square" alt="Transactions">
  <img src="https://img.shields.io/badge/Fraud%20Cases-492-b42318?style=flat-square" alt="Fraud cases">
  <img src="https://img.shields.io/badge/Fraud%20Rate-0.1727%25-orange?style=flat-square" alt="Fraud rate">
  <img src="https://img.shields.io/badge/Primary%20Metric-AUPRC-6f42c1?style=flat-square" alt="AUPRC">
</p>

</div>

Overview

Credit-card fraud detection is an extreme class-imbalance problem. In this dataset, only 492 of 284,807 transactions are fraudulent, so a classifier that predicts every transaction as legitimate would still appear highly accurate.

This project therefore treats fraud detection as a rare-event classification problem and prioritizes AUPRC, recall, precision, and F1-score over raw accuracy.

The repository is structured as a reproducible workflow:

EDA → leakage-safe preprocessing → class-aware modeling → threshold tuning → final evaluation

<p align="center">
  <img src="figures/pipeline.png" width="94%" alt="Fraud detection pipeline">
</p>

Dataset

The dataset contains 284,807 transactions and 31 columns. The existing EDA found no missing values and 1,081 duplicated rows before optional deduplication.

<p align="center">
  <img src="figures/dataset_overview.png" width="82%" alt="Dataset overview">
</p>

Feature group

Description

Time

Seconds elapsed between each transaction and the first transaction

V1–V28

PCA-transformed numerical features

Amount

Transaction amount

Class

Target label: 0 legitimate, 1 fraud

The raw CSV is not stored in Git. See data/README.md for setup instructions.

The Core Challenge: 0.17% Fraud

Fraud represents only 0.1727% of all transactions.

<p align="center">
  <img src="figures/class_imbalance.png" width="68%" alt="Extreme class imbalance">
</p>

This changes how the project should be designed:

Accuracy is not a useful primary metric.

Training and evaluation splits must preserve the minority-class ratio.

Resampling, if used, must be applied only to training data.

Threshold selection should be performed on validation data, never on the test set.

The final model must balance missed fraud (false negatives) against unnecessary alerts (false positives).

Data Quality & EDA

The current EDA notebook verifies:

Check

Finding

Dataset shape

284,807 × 31

Missing values

0

Duplicate rows

1,081

Legitimate transactions

284,315

Fraudulent transactions

492

Fraud share

0.1727%

<p align="center">
  <img src="figures/class_share.png" width="68%" alt="Class share">
</p>

The EDA also explores transaction amount, time, PCA-derived variables, and feature correlations before any modeling decision is made.

Preprocessing Strategy

The preprocessing module is designed to avoid the most common leakage errors in fraud detection.

1. Deduplication

Duplicate rows can optionally be removed before splitting.

2. Stratified splitting

The data is divided into train / validation / test sets using the class label for stratification so the rare fraud proportion is represented in every split.

3. Training-only scaling

Time and Amount are scaled using RobustScaler, with the scaler fit only on the training split and then applied to validation and test data.

The PCA-derived V1–V28 variables are left unchanged.

Modeling

The repository provides two class-aware baselines:

Model

Imbalance handling

Role

Logistic Regression

class_weight="balanced"

Interpretable baseline

Random Forest

Balanced subsample weights

Nonlinear ensemble baseline

The implementation deliberately keeps validation and test sets in their natural imbalanced distribution.

Model scores are not hard-coded into this README. Running the pipeline generates artifacts/metrics.json, ensuring displayed results can be traced to an actual experiment rather than copied or fabricated.

Evaluation

The evaluation module reports metrics designed for rare-event detection:

AUPRC

Area Under the Precision–Recall Curve is the primary ranking metric because it focuses directly on performance for the positive minority class.

Recall

Measures how many real fraud cases are successfully detected.

Precision

Measures how many transactions flagged as fraud are actually fraudulent.

F1-score

Balances precision and recall at a chosen decision threshold.

ROC-AUC

Reported as a secondary discrimination metric, but not used alone because ROC-AUC can look optimistic under extreme class imbalance.

Threshold Tuning

A default threshold of 0.5 is rarely the only sensible operating point for a fraud system.

The pipeline:

fits the classifier on training data;

predicts probabilities on the validation set;

selects a threshold using validation performance;

evaluates that fixed threshold once on the untouched test set.

This separates model discrimination from the business decision rule used to turn probabilities into alerts.

Repository Structure

credit-card-fraud-detection/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── figures/
│   ├── class_imbalance.png
│   ├── class_share.png
│   ├── dataset_overview.png
│   └── pipeline.png
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_evaluation.py
├── artifacts/
└── models/

The existing 01_eda.ipynb should be retained from the original repository; the package supplied here replaces the previously empty downstream notebooks with runnable workflow notebooks.

Quick Start

python -m venv .venv

Activate the environment, then install dependencies:

pip install -r requirements.txt

Place creditcard.csv under data/, then run:

python -m src.train --data data/creditcard.csv

The training run writes test metrics to:

artifacts/metrics.json

and fitted estimators to:

models/

Run the tests with:

pytest -q

Design Principles

Prevent leakage. Scaling and any future resampling must be learned from training data only.

Evaluate the minority class directly. AUPRC and recall matter more than headline accuracy.

Tune thresholds separately from training. The operating point is a business decision, not just a model default.

Keep the test set untouched. Test performance should represent a final unbiased estimate.

Prefer reproducible metrics over impressive-looking numbers. Results belong in the README only after a recorded run generates them.

Next Improvements

Add SMOTE and undersampling as training-only comparison arms.

Add XGBoost or LightGBM with imbalance-aware weighting.

Use StratifiedKFold for more stable model comparison.

Add calibration analysis for predicted fraud probabilities.

Compare threshold policies using explicit false-negative and false-positive costs.

Generate PR curves and confusion-matrix figures automatically from metrics.json.

Add GitHub Actions for unit tests.

<div align="center">

Credit Card Fraud Detection

Rare-Event Classification · Imbalanced Learning · Threshold Optimization

</div>
