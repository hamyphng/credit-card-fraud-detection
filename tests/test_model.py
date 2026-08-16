import pandas as pd
from sklearn.datasets import make_classification

from src.model import (
    build_class_weighted_logistic,
    get_baseline_models,
    load_model,
    save_model,
)


def test_baseline_model_registry_contains_expected_models():
    assert set(get_baseline_models()) == {
        "class_weighted_logistic",
        "smote_logistic",
        "undersampled_logistic",
        "class_weighted_random_forest",
        "xgboost",
        "lightgbm",
    }


def test_saved_model_can_be_loaded_and_used(tmp_path):
    X_values, y = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.8, 0.2],
        random_state=42,
    )
    X = pd.DataFrame(
        X_values,
        columns=["Time", "Amount", "V1", "V2", "V3", "V4"],
    )
    model = build_class_weighted_logistic().fit(X, y)
    filepath = tmp_path / "model.joblib"

    save_model(model, filepath)
    loaded_model = load_model(filepath)

    assert loaded_model.predict(X).shape == y.shape
    assert "preprocessor" in loaded_model.named_steps
