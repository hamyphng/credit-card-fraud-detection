# Credit Card Fraud Detection

A machine-learning project for detecting fraudulent credit-card transactions. The dataset is highly imbalanced, so the project focuses on metrics such as recall, precision, F1-score, and AUPRC instead of accuracy alone.

## Project structure

```text
finance/
├── data/                 # Local dataset directory (not committed to GitHub)
├── notebooks/            # Exploratory analysis, preprocessing, modeling, evaluation
├── src/                  # Reusable Python modules
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── plan.md               # Project roadmap (Vietnamese)
└── README.md
```

## Requirements

- Python 3.11.9
- pip

## Setup

From the project directory:

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Dataset

Download `creditcard.csv` and place it here:

```text
data/creditcard.csv
```

The dataset file is intentionally excluded from Git because it is larger than GitHub's 100 MB size limit. This project expects the Kaggle credit-card fraud dataset, with the columns `Time`, `V1` through `V28`, `Amount`, and `Class`.

## Run the EDA notebook

1. Open `notebooks/01_eda.ipynb` in VS Code.
2. Select the `.venv` Python 3.11.9 kernel.
3. Run the cells in order.

The notebook checks data quality and class imbalance, then explores transaction amount, time, PCA features, and feature correlations.

## Development roadmap

1. Explore the data in `notebooks/01_eda.ipynb`.
2. Scale features and create stratified train/validation/test splits.
3. Compare imbalance-handling approaches such as class weighting and SMOTE.
4. Train baseline and ensemble models.
5. Evaluate with recall, precision, F1-score, and AUPRC; tune the classification threshold.
6. Refactor reusable code into `src/` and add unit tests.

See [plan.md](plan.md) for the detailed project plan.

## Testing

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
```

## Notes

- Never apply oversampling or undersampling to validation or test data; this causes data leakage.
- Use `stratify=y` when splitting data to preserve the fraud ratio.
- Accuracy is misleading for this heavily imbalanced dataset.
