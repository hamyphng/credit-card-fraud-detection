import numpy as np
import pandas as pd

from src.preprocessing import SCALE_COLUMNS, split_and_scale_data, split_data


def make_transactions() -> pd.DataFrame:
    rows = 100
    return pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            "Amount": np.linspace(1, 1000, rows),
            "V1": np.linspace(-2, 2, rows),
            "Class": [0] * 80 + [1] * 20,
        }
    )


def test_split_and_scale_removes_duplicates_and_preserves_class_ratio():
    df = make_transactions()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale_data(df)

    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert (len(X_train), len(X_val), len(X_test)) == (60, 20, 20)
    assert y_train.mean() == y_val.mean() == y_test.mean() == 0.2
    assert list(scaler.feature_names_in_) == SCALE_COLUMNS


def test_scaler_is_fit_on_train_only():
    df = make_transactions()
    X_train, _, _, _, _, _, scaler = split_and_scale_data(df)

    original_train_values = df.loc[X_train.index, SCALE_COLUMNS]

    assert np.allclose(scaler.center_, original_train_values.median().to_numpy())


def test_split_data_keeps_raw_time_and_amount_values():
    df = make_transactions()
    X_train, _, _, _, _, _ = split_data(df)

    assert X_train["Time"].max() > 1
    assert X_train["Amount"].max() > 1
