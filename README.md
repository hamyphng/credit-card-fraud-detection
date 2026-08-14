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
focused on understanding fraudulent behavior and establishing a reliable
foundation for fraud-detection modeling.
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
| **Primary Challenge** | Extreme class imbalance |

> **Headline finding:** fewer than **2 in every 1,000 transactions** are fraudulent. A naïve classifier could exceed 99.8% accuracy while detecting no fraud at all.

---

## Navigation

<p align="center">
  <a href="#dataset">Dataset</a> •
  <a href="#exploratory-data-analysis">EDA</a> •
  <a href="#key-discovery">Key Discovery</a> •
  <a href="#feature-analysis">Features</a> •
  <a href="#modeling-strategy">Modeling</a> •
  <a href="#evaluation-strategy">Evaluation</a> •
  <a href="#key-findings">Findings</a>
</p>

---

## Dataset

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

However, **1,081 duplicated rows** were identified during exploratory analysis. These observations should be investigated or removed consistently before model development.

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

### Only 0.1727% of the dataset

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

### Important Interpretation Constraint

`V1–V28` should be interpreted **statistically rather than semantically**.

For example, observing that `V14` differs strongly between fraudulent and legitimate transactions could make it useful for prediction, but it would not justify assigning a specific real-world meaning to `V14`.

---

# Modeling Strategy

## From exploration to fraud detection

The planned modeling workflow is:

<div align="center">

**Raw Transactions**

↓

**Data Quality & Deduplication**

↓

**Stratified Train / Validation / Test Split**

↓

**Feature Preprocessing**

↓

**Imbalance-Aware Classification**

↓

**Threshold Optimization**

↓

**Final Test Evaluation**

</div>

### 01 — Prepare

Investigate duplicate observations and establish a consistent preprocessing policy.

### 02 — Split

Create stratified training, validation, and test sets so the rare fraud class remains represented across all partitions.

### 03 — Transform

Scale `Time` and `Amount` using parameters learned from the training data only.

### 04 — Model

Compare interpretable baseline models with nonlinear and imbalance-aware classifiers.

### 05 — Tune

Select the classification threshold using validation data instead of automatically relying on `0.5`.

### 06 — Evaluate

Evaluate the selected model once on the untouched test set.

---

## Candidate Models

| Model | Role |
|---|---|
| **Logistic Regression** | Interpretable baseline |
| **Random Forest** | Nonlinear ensemble baseline |
| **Gradient Boosting** | Capture complex feature interactions |
| **Class-Weighted Models** | Increase attention to rare fraud cases |
| **Resampling Strategies** | Improve minority-class learning during training |

> Final model scores are intentionally not reported yet because the current repository contains completed exploratory analysis but does not yet contain a verified end-to-end modeling experiment.

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
| **ROC-AUC** | How well does the model separate the two classes? |

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

The optimal classification threshold therefore depends on the relative cost of these two errors.

---

# Key Findings

<table>
<tr>
<td width="50%" valign="top">

### Extreme class imbalance

Only **0.1727%** of transactions are fraudulent.

The minority class is approximately **1 in every 578 transactions**.

</td>
<td width="50%" valign="top">

### Accuracy is deceptive

A model could exceed **99.8% accuracy** while detecting absolutely no fraud.

</td>
</tr>

<tr>
<td width="50%" valign="top">

### Data quality is strong

The dataset contains **no missing values**, although **1,081 duplicate rows** require attention.

</td>
<td width="50%" valign="top">

### Evaluation design matters

AUPRC, recall, precision, and F1-score provide substantially more meaningful information than accuracy alone.

</td>
</tr>
</table>

---

# Interpretation

The exploratory analysis shows that credit-card fraud detection should be treated as a **rare-event classification problem**, not a conventional balanced machine-learning task.

The extreme class imbalance means that overall correctness provides little information about whether a model can identify fraudulent activity.

A reliable modeling pipeline should therefore:

- preserve realistic class distributions during evaluation;
- prevent preprocessing leakage;
- evaluate minority-class performance directly;
- separate probability estimation from threshold selection;
- analyze both false negatives and false positives.

The anonymized PCA features may provide substantial predictive value, but predictive usefulness should not be confused with direct business interpretation.

---

# Limitations

- `V1–V28` are anonymized, limiting direct feature interpretation.
- The dataset represents historical transactions and may not capture future fraud patterns.
- Extreme class imbalance makes performance sensitive to metric selection.
- Duplicate observations require a consistent preprocessing decision.
- The current project version does not yet contain a verified final classifier.
- Real-world deployment would additionally require monitoring for concept drift.
- Business costs associated with false negatives and false positives are unavailable.

---

## Next Steps

The next stage of the project will focus on turning the exploratory analysis into a reproducible fraud-detection experiment:

**01.** Establish the duplicate-handling policy  
**02.** Create leakage-safe stratified data splits  
**03.** Build interpretable baseline classifiers  
**04.** Compare imbalance-handling strategies  
**05.** Evaluate using AUPRC, precision, recall, and F1-score  
**06.** Optimize the decision threshold  
**07.** Analyze false positives and false negatives  
**08.** Report final performance on an untouched test set

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
└── tests/
