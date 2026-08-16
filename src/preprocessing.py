import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


SCALE_COLUMNS = ["Time", "Amount"]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
):
    """Remove duplicates and create stratified raw train/validation/test splits."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < validation_size < 1 - test_size:
        raise ValueError("validation_size must be positive and leave room for training data")

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

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_and_scale_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
):
    """Return stratified splits with scaling fitted on the full training split.

    Use this helper for a final train/validation/test workflow. For
    cross-validation, pass raw splits from :func:`split_data` to a model
    pipeline that contains its own preprocessor for every CV fold.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    scaler = RobustScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[SCALE_COLUMNS] = scaler.fit_transform(X_train[SCALE_COLUMNS])
    X_val[SCALE_COLUMNS] = scaler.transform(X_val[SCALE_COLUMNS])
    X_test[SCALE_COLUMNS] = scaler.transform(X_test[SCALE_COLUMNS])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
