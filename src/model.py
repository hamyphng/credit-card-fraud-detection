"""Baseline models for credit-card fraud detection."""

from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42


def build_class_weighted_logistic() -> LogisticRegression:
    """Create a baseline that compensates for the minority class with weights."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )


def build_smote_logistic() -> Pipeline:
    """Create a SMOTE + Logistic Regression pipeline safe for cross-validation."""
    return Pipeline(
        steps=[
            (
                "sampler",
                SMOTE(
                    sampling_strategy=0.1,
                    random_state=RANDOM_STATE,
                    k_neighbors=5,
                ),
            ),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def build_undersampled_logistic() -> Pipeline:
    """Create an undersampling + Logistic Regression pipeline safe for CV."""
    return Pipeline(
        steps=[
            (
                "sampler",
                RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE),
            ),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def build_class_weighted_random_forest() -> RandomForestClassifier:
    """Create a tree-based baseline that accounts for the minority class."""
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def build_xgboost() -> XGBClassifier:
    """Create an imbalance-aware XGBoost classifier for tabular data."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=100,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )


def build_lightgbm() -> LGBMClassifier:
    """Create an imbalance-aware LightGBM classifier for tabular data."""
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )


def get_baseline_models() -> dict:
    """Return the models used for cross-validation comparison."""
    return {
        "class_weighted_logistic": build_class_weighted_logistic(),
        "smote_logistic": build_smote_logistic(),
        "undersampled_logistic": build_undersampled_logistic(),
        "class_weighted_random_forest": build_class_weighted_random_forest(),
        "xgboost": build_xgboost(),
        "lightgbm": build_lightgbm(),
    }


def save_model(model: object, filepath: str | Path) -> None:
    """Persist a fitted model to disk."""
    joblib.dump(model, filepath)


def load_model(filepath: str | Path) -> object:
    """Load a model saved with :func:`save_model`."""
    return joblib.load(filepath)
