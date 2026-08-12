import pandas as pd
import os

def load_data(filepath: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load raw data from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    df = pd.read_csv(filepath)
    print("Successfully loaded dataset")
    return df
    