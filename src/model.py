"""Baseline models for credit-card fraud detection."""

from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
SCALE_COLUMNS = ["Time", "Amount"]


def build_preprocessor() -> ColumnTransformer:
    """Scale only raw Time and Amount columns inside each model fit."""
    return ColumnTransformer(
        transformers=[("robust_scale", RobustScaler(), SCALE_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(model: object, sampler: object | None = None) -> Pipeline:
    """Create a leakage-safe preprocessing/model pipeline for cross-validation."""
    steps: list[tuple[str, object]] = [("preprocessor", build_preprocessor())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def build_class_weighted_logistic() -> Pipeline:
    """Create a baseline that compensates for the minority class with weights."""
    return build_model_pipeline(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    )


def build_smote_logistic() -> Pipeline:
    """Create a SMOTE + Logistic Regression pipeline safe for cross-validation."""
    return build_model_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        sampler=SMOTE(
            sampling_strategy=0.1,
            random_state=RANDOM_STATE,
            k_neighbors=5,
        ),
    )


def build_undersampled_logistic() -> Pipeline:
    """Create an undersampling + Logistic Regression pipeline safe for CV."""
    return build_model_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        sampler=RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE),
    )


def build_class_weighted_random_forest() -> Pipeline:
    """Create a tree-based baseline that accounts for the minority class."""
    return build_model_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    )


def build_xgboost() -> Pipeline:
    """Create an imbalance-aware XGBoost classifier for tabular data."""
    return build_model_pipeline(
        XGBClassifier(
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
    )


def build_lightgbm() -> Pipeline:
    """Create an imbalance-aware LightGBM classifier for tabular data."""
    return build_model_pipeline(
        LGBMClassifier(
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
