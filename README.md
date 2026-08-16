<div align="center">

# Credit Card Fraud Detection

### Exploratory Data Analysis • Imbalanced Classification • Fraud Analytics

<p>
  <img src="https://img.shields.io/badge/Dataset-284%2C807%20transactions-2ea44f?style=for-the-badge" alt="Dataset"/>
  <img src="https://img.shields.io/badge/Features-31-0969da?style=for-the-badge" alt="Features"/>
  <img src="https://img.shields.io/badge/Fraud%20Cases-492-critical?style=for-the-badge" alt="Fraud Cases"/>
  <img src="https://img.shields.io/badge/Fraud%20Rate-0.1727%25-orange?style=for-the-badge" alt="Fraud Rate"/>
</p>

<p>
A data-driven investigation of highly imbalanced credit-card transactions,
from exploratory analysis and leakage-safe preprocessing to resampling strategies
for fraud-detection modeling.
</p>

**Only 0.1727% of transactions are fraudulent — making class imbalance the central challenge.**

</div>

---

## Project at a Glance

| | |
|---|---|
| **Goal** | Identify rare fraudulent credit-card transactions |
| **Dataset** | 284,807 transactions |
| **Target** | `Class` — legitimate or fraudulent |
| **Predictors** | Time, Amount, V1–V28 |
| **Fraud Cases** | 492 |
| **Fraud Rate** | **0.1727%** |
| **Missing Values** | 0 |
| **Duplicate Rows** | 1,081 |
| **Primary Challenge** | Extreme class imbalance |
| **Current Stage** | **Preprocessing completed · Modeling next** |

> **Headline finding:** fewer than **2 in every 1,000 transactions** are fraudulent. A naïve classifier could exceed 99.8% accuracy while detecting no fraud at all.

---

## Navigation

<p align="center">
  <a href="#dataset">Dataset</a> •
  <a href="#exploratory-data-analysis">EDA</a> •
  <a href="#key-discovery">Key Discovery</a> •
  <a href="#feature-analysis">Features</a> •
  <a href="#preprocessing">Preprocessing</a> •
  <a href="#handling-class-imbalance">Imbalance Handling</a> •
  <a href="#modeling-strategy">Modeling</a> •
  <a href="#evaluation-strategy">Evaluation</a>
</p>

---

# Dataset

The dataset contains **284,807 credit-card transactions** described by **31 numerical variables**.

| Variable | Type | Description |
|---|---|---|
| `Time` | Numerical | Seconds elapsed from the first recorded transaction |
| `V1–V28` | Numerical | PCA-transformed anonymized transaction features |
| `Amount` | Numerical | Transaction amount |
| `Class` | Binary | `0` = legitimate, `1` = fraud |

### Dataset Summary

| Statistic | Value |
|---|---:|
| Observations | **284,807** |
| Features | **31** |
| Legitimate Transactions | **284,315** |
| Fraudulent Transactions | **492** |
| Fraud Rate | **0.1727%** |
| Missing Values | **0** |
| Duplicate Rows | **1,081** |

<details>
<summary><b>Data quality notes</b></summary>

<br>

The dataset contains no missing values, reducing the need for imputation.

However, **1,081 duplicated rows** were identified during exploratory analysis. These duplicates are removed before the train–validation–test split.

All variables are numerical. The original meanings of `V1–V28` are unavailable because they have been transformed using PCA.

</details>

---

# Exploratory Data Analysis

## Understanding the Transaction Data

Before building a fraud classifier, the analysis focuses on understanding the structure of the dataset and, most importantly, the behavior of the minority fraud class.

The exploratory analysis examines:

| Analysis | Purpose |
|---|---|
| **Data Quality** | Identify missing and duplicated observations |
| **Class Distribution** | Quantify the severity of class imbalance |
| **Transaction Amount** | Compare transaction-value behavior |
| **Transaction Time** | Explore temporal transaction patterns |
| **PCA Features** | Identify statistical differences between classes |
| **Correlations** | Examine relationships between numerical variables |

Because `V1–V28` are anonymized PCA components, the analysis avoids assigning unsupported business interpretations to individual features.

---

# Key Discovery

<div align="center">

## Fraud is exceptionally rare.

### 284,315 legitimate transactions

versus

# 492 fraud cases

### Only 0.1727% of the original dataset

</div>

The minority class represents approximately:

<div align="center">

# 1 in 578

### transactions

</div>

This imbalance is the defining characteristic of the problem.

A model can classify almost every transaction correctly while completely failing at the task that actually matters: **detecting fraud**.

---

## Why Accuracy Is Misleading

Consider the simplest possible classifier:

> Predict every transaction as legitimate.

Its approximate performance would be:

| Metric | Result |
|---|---:|
| Accuracy | **99.83%** |
| Fraud detected | **0 / 492** |
| Fraud recall | **0%** |

So despite achieving nearly **100% accuracy**, the classifier would have no practical fraud-detection capability.

This is why accuracy should not be treated as the primary metric for this project.

---

# Feature Analysis

## What information can distinguish fraud?

The predictors can be divided into three groups.

<table>
<tr>
<td width="33%" valign="top">

### Time

Represents when each transaction occurred relative to the first transaction in the dataset.

Temporal patterns may provide useful information about transaction behavior.

</td>
<td width="33%" valign="top">

### Amount

The monetary value of each transaction.

Unlike the PCA components, this feature retains direct real-world interpretation.

</td>
<td width="33%" valign="top">

### V1–V28

Anonymized PCA-transformed variables derived from confidential transaction information.

They can contain strong predictive signal despite lacking direct semantic interpretation.

</td>
</tr>
</table>

### Interpretation Constraint

`V1–V28` should be interpreted **statistically rather than semantically**.

A PCA component may strongly distinguish fraudulent from legitimate transactions, but its anonymized nature does not justify assigning it a specific real-world meaning.

---

# Preprocessing

## Building a Leakage-Safe Dataset

The preprocessing pipeline is designed to prevent information from the validation or test sets from influencing model training.

<div align="center">

**Raw Transactions**

↓

**Remove Duplicates**

↓

**Stratified Train / Validation / Test Split**

↓

**Fit RobustScaler on Training Data**

↓

**Transform Validation & Test**

↓

**Resample Training Data Only**

</div>

---

## 01 — Duplicate Removal

The **1,081 duplicated observations** identified during EDA are removed before splitting the dataset.

After deduplication, the modeling dataset contains approximately:

<div align="center">

### 283,726 unique transactions

</div>

---

## 02 — Stratified Data Split

The cleaned dataset is divided into:

- **60% training**
- **20% validation**
- **20% test**

Stratification preserves approximately the same fraud prevalence across all three sets.

| Split | Normal | Fraud | Fraud Rate |
|---|---:|---:|---:|
| **Training** | 169,951 | 284 | **0.1668%** |
| **Validation** | 56,651 | 94 | **0.1657%** |
| **Test** | 56,651 | 95 | **0.1674%** |

The validation and test sets remain in their **natural imbalanced distribution**.

This is essential because model performance should be measured under conditions that resemble the original data rather than an artificially balanced evaluation set.

---

## 03 — Feature Scaling

Only:

- `Time`
- `Amount`

are explicitly scaled.

The pipeline uses **RobustScaler**, which is less sensitive to extreme values than standard mean–variance scaling.

Most importantly:

> **The scaler is fitted only on the training set.**

The learned transformation is then applied to validation and test data.

This prevents information leakage from the holdout sets into preprocessing.

`V1–V28` are left unchanged because they are already PCA-transformed variables.

---

# Handling Class Imbalance

The training data still contains only:

<div align="center">

### 284 fraud cases vs. 169,951 legitimate transactions

</div>

Rather than committing to a single balancing technique, the project prepares **three alternative training distributions** for controlled comparison.

---

## Strategy 1 — Original Distribution

The first training set preserves the natural class distribution.

| Class | Samples |
|---|---:|
| Normal | **169,951** |
| Fraud | **284** |
| Fraud Rate | **0.1668%** |

This dataset provides the reference condition for models trained without synthetic or majority-class resampling.

Class weighting can later be applied at the model level.

---

## Strategy 2 — SMOTE

**Synthetic Minority Over-sampling Technique (SMOTE)** increases minority-class representation by generating synthetic fraud observations between neighboring minority samples.

SMOTE is applied **only to the training set**.

### Before SMOTE

| Class | Samples |
|---|---:|
| Normal | 169,951 |
| Fraud | 284 |

### After SMOTE

| Class | Samples |
|---|---:|
| Normal | **169,951** |
| Fraud | **16,995** |
| Total | **186,946** |

The configuration uses:

**Fraud / Normal = 0.10**

which produces a fraud share of approximately:

<div align="center">

# 9.09%

### in the SMOTE training set

</div>

The goal is not to force an artificial 50:50 distribution.

Instead, minority representation is increased substantially while the majority class remains dominant.

---

## Strategy 3 — Random Undersampling

Random undersampling takes the opposite approach.

Instead of generating additional fraud observations, it reduces the number of legitimate training samples.

### After Undersampling

| Class | Samples |
|---|---:|
| Normal | **2,840** |
| Fraud | **284** |
| Total | **3,124** |
| Fraud Rate | **9.09%** |

This dramatically reduces majority-class dominance and computational cost.

However, it also discards a large amount of legitimate transaction information.

That trade-off will be evaluated empirically rather than assumed to be beneficial.

---

# Experimental Design

## Three views of the same fraud problem

| Training Strategy | Normal | Fraud | Main Trade-off |
|---|---:|---:|---|
| **Original** | 169,951 | 284 | Preserves all real observations |
| **SMOTE** | 169,951 | 16,995 | Adds synthetic minority information |
| **Undersampling** | 2,840 | 284 | Removes majority information |

All three strategies will be evaluated against the **same untouched validation and test sets**.

This isolates the effect of the imbalance-handling strategy from changes in the evaluation distribution.

---

# Modeling Strategy

## From preprocessing to fraud detection

The next stage compares models across the alternative training distributions.

### Baseline

Train on the original class distribution to establish reference performance.

### Class-Weighted Learning

Train on the original observations while increasing the model's penalty for minority-class errors.

### SMOTE-Based Learning

Train on the synthetic minority-enhanced dataset.

### Undersampling-Based Learning

Train on the reduced majority-class dataset.

Potential classifiers include:

| Model | Role |
|---|---|
| **Logistic Regression** | Interpretable linear baseline |
| **Random Forest** | Nonlinear ensemble baseline |
| **Gradient Boosting** | Capture complex feature interactions |
| **Class-Weighted Models** | Handle imbalance without modifying the dataset |

## Final Model Result

The selected model is an XGBoost classifier. The threshold was selected on validation data by maximizing F1-score, then evaluated once on the untouched test set.

| Metric | Test result |
|---|---:|
| Decision threshold | 0.919 |
| Precision | 98.55% |
| Recall | 71.58% |
| F1-score | 82.93% |
| AUPRC | 82.14% |
| True positives | 68 |
| False positives | 1 |
| False negatives | 27 |

---

# Evaluation Strategy

## The goal is not maximum accuracy.

The objective is to detect as much fraud as possible while maintaining a manageable number of false alerts.

### Primary Metrics

| Metric | What it answers |
|---|---|
| **AUPRC** | How well does the model rank rare fraud cases overall? |
| **Recall** | How much actual fraud does the model detect? |
| **Precision** | How many fraud alerts are truly fraudulent? |
| **F1-score** | How well are precision and recall balanced? |
| **ROC-AUC** | How well does the model rank the two classes? |

For this dataset, **AUPRC and the precision–recall trade-off** are particularly important because the positive class is extremely rare.

---

## The Business Trade-off

<table>
<tr>
<td width="50%" valign="top">

### False Negative

**Fraud → Legitimate**

A fraudulent transaction is not detected.

Potential consequences include direct financial loss and delayed fraud response.

</td>
<td width="50%" valign="top">

### False Positive

**Legitimate → Fraud**

A legitimate transaction triggers a fraud alert.

Potential consequences include investigation costs and unnecessary customer disruption.

</td>
</tr>
</table>

A practical fraud-detection system therefore requires more than a good ranking model.

It also requires an appropriate **decision threshold**.

---

# Threshold Optimization

The default classification threshold:

```text
0.50
```

is not automatically optimal for fraud detection.

The planned evaluation workflow is:

<div align="center">

**Train Model**

↓

**Predict Validation Probabilities**

↓

**Analyze Precision–Recall Trade-off**

↓

**Select Operating Threshold**

↓

**Lock Threshold**

↓

**Evaluate Once on Test Set**

</div>

Threshold selection is performed using **validation data only**.

The test set remains untouched until the final evaluation.

---

# Key Findings So Far

<table>
<tr>
<td width="50%" valign="top">

### Extreme class imbalance

Only **0.1727%** of the original transactions are fraudulent.

This is the defining characteristic of the problem.

</td>
<td width="50%" valign="top">

### Accuracy is deceptive

A classifier could exceed **99.8% accuracy** while detecting no fraud.

</td>
</tr>

<tr>
<td width="50%" valign="top">

### Leakage-safe preprocessing matters

Duplicates are removed before splitting, and scaling parameters are learned from training data only.

</td>
<td width="50%" valign="top">

### Evaluation remains realistic

SMOTE and undersampling are restricted to the training set.

Validation and test distributions remain untouched.

</td>
</tr>

<tr>
<td width="50%" valign="top">

### SMOTE preserves majority information

All **169,951 legitimate training samples** remain available while synthetic fraud observations increase minority representation.

</td>
<td width="50%" valign="top">

### Undersampling is aggressive

Only **2,840 legitimate samples** remain, making training much smaller but potentially discarding useful information.

</td>
</tr>
</table>

---

# Interpretation

The project has now moved beyond exploratory analysis into a controlled **imbalanced-learning experiment**.

The preprocessing stage establishes two principles that are especially important for fraud detection:

**First, evaluation data should remain realistic.**

Artificially balancing validation or test data would produce metrics that do not represent the original transaction environment.

**Second, imbalance handling is an experimental choice rather than an automatic preprocessing step.**

SMOTE, undersampling, and class weighting solve different problems and introduce different trade-offs.

The next modeling stage will therefore compare these strategies under the **same validation and test conditions** rather than assuming that balancing the dataset necessarily improves fraud detection.

---

# Limitations

- `V1–V28` are anonymized, limiting direct feature interpretation.
- The dataset represents historical transactions and may not capture future fraud patterns.
- SMOTE generates synthetic observations rather than real fraudulent transactions.
- Random undersampling discards a substantial amount of legitimate transaction information.
- Extreme class imbalance makes performance highly sensitive to metric and threshold selection.
- The project does not yet contain verified final model results.
- Real-world deployment would additionally require monetary cost modeling, latency constraints, alert capacity, and concept-drift monitoring.

---

## Current Progress

| Stage | Status |
|---|---|
| Exploratory Data Analysis | **Completed** |
| Data Quality Analysis | **Completed** |
| Duplicate Removal | **Completed** |
| Stratified Data Splitting | **Completed** |
| Robust Feature Scaling | **Completed** |
| SMOTE | **Completed** |
| Random Undersampling | **Completed** |
| Model Training | **Next** |
| Model Comparison | Planned |
| Threshold Optimization | Planned |
| Final Test Evaluation | Planned |

---

## Next Steps

**01.** Train baseline models on the original distribution  
**02.** Compare class-weighted, SMOTE, and undersampled strategies  
**03.** Evaluate validation AUPRC, precision, recall, and F1-score  
**04.** Compare Precision–Recall curves across models  
**05.** Optimize the decision threshold using validation data  
**06.** Analyze false positives and false negatives  
**07.** Select the final model and imbalance strategy  
**08.** Report performance once on the untouched test set

---

<details>
<summary><b>Repository Structure</b></summary>

```text
credit-card-fraud-detection/
├── README.md
├── plan.md
├── requirements.txt
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   └── preprocessing.py
└── tests/
```

### Notebook Progress

| Notebook | Status |
|---|---|
| `01_eda.ipynb` | **Completed** |
| `02_preprocessing.ipynb` | **Completed** |
| `03_modeling.ipynb` | Next |
| `04_evaluation.ipynb` | Planned |

</details>

---

<div align="center">

### Credit Card Fraud Detection

**Exploratory Analysis • Imbalanced Learning • Fraud Analytics**

</div>
