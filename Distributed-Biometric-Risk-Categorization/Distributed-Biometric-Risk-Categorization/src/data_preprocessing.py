"""Data loading, inspection, cleaning and splitting.

Missing-value imputation is intentionally NOT done here: it lives inside the
scikit-learn pipeline (see feature_engineering.py) so that exactly the same
imputation is applied at training time and at prediction time.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE


def load_data(path):
    """Load a CSV file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Pass your CSV with --data "
            "(or use --demo / data/sample/sample_data.csv for the demo data)."
        )
    return pd.read_csv(path)


def inspect_data(df, target_column=None):
    """Return a basic summary of the dataset."""
    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if target_column is not None and target_column in df.columns:
        summary["class_distribution"] = df[target_column].value_counts(dropna=False).to_dict()
    return summary


def clean_data(df, target_column, drop_duplicates=False):
    """Drop rows with a missing target; optionally drop exact duplicate rows."""
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"
        )
    cleaned = df.dropna(subset=[target_column])
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    return cleaned.reset_index(drop=True)


def split_features_target(df, target_column):
    """Separate the feature matrix X from the target vector y."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def split_train_test(X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
    """Stratified, reproducible train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
