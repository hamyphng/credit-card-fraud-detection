"""Evaluation helpers for binary fraud-detection models."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def evaluate_at_threshold(
    y_true: pd.Series,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate fraud metrics after converting probabilities to labels."""
    y_pred = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auprc": average_precision_score(y_true, y_probability),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def evaluate_thresholds(
    y_true: pd.Series,
    y_probability: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Return comparable metrics for a list of decision thresholds."""
    return pd.DataFrame(
        [evaluate_at_threshold(y_true, y_probability, threshold) for threshold in thresholds]
    )


def find_best_f1_threshold(
    y_true: pd.Series,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Find the precision-recall threshold that maximizes validation F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probability)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.argmax(f1_scores))

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_scores[best_index]),
    }
