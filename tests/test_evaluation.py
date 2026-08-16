import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    evaluate_at_threshold,
    evaluate_thresholds,
    find_best_f1_threshold,
)


def test_evaluate_at_threshold_returns_expected_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.3, 0.6, 0.9])

    metrics = evaluate_at_threshold(y_true, probability, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0


def test_threshold_helpers_return_comparable_results():
    y_true = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.3, 0.6, 0.9])

    comparison = evaluate_thresholds(y_true, probability, np.array([0.3, 0.5]))
    best = find_best_f1_threshold(y_true, probability)

    assert comparison.shape == (2, 9)
    assert 0 <= best["threshold"] <= 1
    assert best["f1"] == pytest.approx(1.0)
