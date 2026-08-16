"""Run the leakage-safe model-selection and final-evaluation workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_data
from src.evaluation import evaluate_at_threshold, evaluate_thresholds, find_best_f1_threshold
from src.model import build_xgboost, get_baseline_models
from src.preprocessing import split_data


RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_data(str(ROOT / "data" / "creditcard.csv"))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1", "auprc": "average_precision"}
    rows = []
    for name, model in get_baseline_models().items():
        print(f"Evaluating {name}...", flush=True)
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({"model": name, **{metric: scores[f"test_{metric}"].mean() for metric in scoring}})
    pd.DataFrame(rows).sort_values("auprc", ascending=False).to_csv(
        RESULTS_DIR / "model_comparison.csv", index=False
    )

    model = build_xgboost().fit(X_train, y_train)
    y_val_probability = model.predict_proba(X_val)[:, 1]
    threshold = find_best_f1_threshold(y_val, y_val_probability)["threshold"]
    evaluate_thresholds(y_val, y_val_probability, np.linspace(0.01, 0.99, 99)).to_csv(
        RESULTS_DIR / "threshold_analysis.csv", index=False
    )

    y_test_probability = model.predict_proba(X_test)[:, 1]
    final_metrics = evaluate_at_threshold(y_test, y_test_probability, threshold=threshold)
    (RESULTS_DIR / "final_test_metrics.json").write_text(
        json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
