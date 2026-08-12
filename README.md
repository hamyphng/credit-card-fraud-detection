<div align="center">

# Credit Card Fraud Detection

### Exploratory Analysis of Highly Imbalanced Transaction Data

Understanding the structure, imbalance, and statistical characteristics of fraudulent credit-card transactions before building predictive models.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-EDA%20Complete-2ea44f?style=flat-square)

</div>

---

## Overview

Credit-card fraud detection is a rare-event classification problem in which fraudulent transactions represent only a tiny fraction of all observations.

This project begins with a detailed exploratory analysis of **284,807 credit-card transactions** to understand the dataset, identify data-quality issues, examine class imbalance, and establish a reliable foundation for subsequent fraud-detection modeling.

### Dataset at a Glance

| | |
|---|---:|
| **Transactions** | 284,807 |
| **Features** | 31 |
| **Fraud Cases** | 492 |
| **Legitimate Cases** | 284,315 |
| **Fraud Rate** | **0.1727%** |
| **Missing Values** | 0 |
| **Duplicate Rows** | 1,081 |

---

## The Problem

### How do we detect fraud when fewer than 2 transactions in 1,000 are fraudulent?

The dataset is extremely imbalanced:

- **284,315** legitimate transactions
- only **492** fraudulent transactions
- fraud accounts for approximately **0.1727%** of the dataset

This means conventional accuracy can be highly misleading.

A classifier predicting every transaction as legitimate would achieve more than **99.8% accuracy** while detecting **zero fraud cases**.

For this reason, the project focuses on understanding the minority class before moving toward metrics such as **precision, recall, F1-score, and AUPRC**.

---

## Dataset

The dataset contains anonymized credit-card transactions represented by 31 numerical variables.

| Feature | Description |
|---|---|
| `Time` | Seconds elapsed between the transaction and the first recorded transaction |
| `V1` – `V28` | PCA-transformed anonymized transaction features |
| `Amount` | Transaction amount |
| `Class` | Target variable: `0` = legitimate, `1` = fraud |

The PCA transformation protects confidential transaction information while retaining statistical patterns useful for analysis.

---

## Exploratory Data Analysis

The current analysis focuses on four questions:

### 01 — Data Quality

Are there missing values, duplicated observations, or structural issues that should be addressed before modeling?

**Findings**

- No missing values were identified.
- **1,081 duplicate rows** were detected.
- The dataset contains only numerical variables.

---

### 02 — Class Distribution

How severe is the imbalance between legitimate and fraudulent transactions?

The imbalance is extreme:

| Class | Transactions | Share |
|---|---:|---:|
| Legitimate | 284,315 | 99.8273% |
| Fraud | 492 | **0.1727%** |

This imbalance is the central modeling challenge of the project.

---

### 03 — Transaction Behavior

How do transaction amount and timing differ across the two classes?

The EDA investigates the distributions of:

- transaction amount;
- transaction time;
- fraud vs. legitimate activity;
- PCA-derived variables;
- feature correlations.

These analyses help identify potentially discriminative patterns before introducing a classification algorithm.

---

### 04 — Feature Relationships

Because `V1`–`V28` are PCA-transformed variables, their original business meaning is unavailable.

Instead of assigning unsupported interpretations to these variables, the analysis focuses on their statistical relationships with the fraud label.

This keeps the analysis grounded in the information actually available in the dataset.

---

## Key Findings

### Extreme imbalance is the defining characteristic

Only **0.1727%** of transactions are fraudulent.

This makes fraud detection fundamentally different from a standard balanced classification task.

### Accuracy alone is inappropriate

A naïve majority-class classifier could exceed **99.8% accuracy** without identifying a single fraudulent transaction.

Evaluation therefore needs to prioritize minority-class performance.

### Data quality is relatively strong

The dataset contains **no missing values**, reducing the need for imputation.

However, **1,081 duplicate observations** should be considered during preprocessing.

### Feature interpretation is intentionally limited

The PCA-derived features provide useful predictive information, but their anonymized nature means they should not be given unsupported real-world interpretations.

---

## Modeling Strategy

The next stage of the project is designed around the characteristics discovered during EDA.

```text
Raw Transactions
       │
       ▼
Data Quality Checks
       │
       ▼
Train / Validation / Test Split
       │
       ▼
Feature Preprocessing
       │
       ▼
Imbalance-Aware Modeling
       │
       ▼
Threshold Optimization
       │
       ▼
Precision · Recall · F1 · AUPRC
