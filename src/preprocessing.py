import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


SCALE_COLUMNS = ["Time", "Amount"]


def split_and_scale_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
):
    """Remove duplicates, stratify splits, and robust-scale Time and Amount.

    The scaler is fitted only on the training set to prevent data leakage.
    """
    clean_df = df.drop_duplicates().copy()
    X = clean_df.drop(columns="Class")
    y = clean_df["Class"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    validation_fraction = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y_train_val,
    )

    scaler = RobustScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[SCALE_COLUMNS] = scaler.fit_transform(X_train[SCALE_COLUMNS])
    X_val[SCALE_COLUMNS] = scaler.transform(X_val[SCALE_COLUMNS])
    X_test[SCALE_COLUMNS] = scaler.transform(X_test[SCALE_COLUMNS])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
