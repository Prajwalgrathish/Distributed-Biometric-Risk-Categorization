"""Train the Random Forest pipeline and save it.

Usage:
    python -m src.train --data path/to/data.csv --target TARGET_COLUMN
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_DATA_PATH,
    DEFAULT_TEST_SIZE,
)
from src.data_preprocessing import (
    clean_data,
    inspect_data,
    load_data,
    split_features_target,
    split_train_test,
)
from src.feature_engineering import build_model_pipeline
from src.model_io import save_model


def resolve_positive_label(classes, requested=None):
    """For binary targets, pick the positive class used by precision/recall/F1.

    Defaults to the last class in sorted order. Returns None for non-binary targets.
    """
    classes = list(classes)
    if len(classes) != 2:
        return None
    if requested is None:
        return classes[-1]
    for cls in classes:
        if str(cls) == str(requested):
            return cls
    raise ValueError(f"--positive-label '{requested}' is not one of the classes: {classes}")


def train_model(X_train, y_train, rf_params=None, random_state=DEFAULT_RANDOM_STATE):
    """Fit preprocessing + Random Forest on the training data."""
    pipeline = build_model_pipeline(X_train, rf_params=rf_params, random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline


def run_training(
    data_path,
    target_column,
    model_path=DEFAULT_MODEL_PATH,
    test_data_path=DEFAULT_TEST_DATA_PATH,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
    rf_params=None,
    positive_label=None,
    drop_duplicates=False,
):
    """Load -> clean -> split -> train -> save model and held-out test set."""
    df = load_data(data_path)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {data_path}")

    summary = inspect_data(df, target_column)
    print(f"Missing values per column: {summary['missing_values']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")

    df = clean_data(df, target_column, drop_duplicates=drop_duplicates)
    X, y = split_features_target(df, target_column)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    pipeline = train_model(X_train, y_train, rf_params, random_state)
    classes = pipeline.classes_.tolist()

    bundle = {
        "pipeline": pipeline,
        "target_column": target_column,
        "feature_columns": X.columns.tolist(),
        "classes": classes,
        "positive_label": resolve_positive_label(classes, positive_label),
        "training_data_path": str(data_path),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "random_state": random_state,
        "rf_params": pipeline.named_steps["classifier"].get_params(),
    }
    save_model(bundle, model_path)

    test_data_path = Path(test_data_path)
    test_data_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_test, y_test], axis=1).to_csv(test_data_path, index=False)

    print(f"Model saved to {model_path}")
    print(f"Held-out test set saved to {test_data_path}")
    return bundle


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the Random Forest risk classifier.")
    parser.add_argument("--data", required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--test-data-path", default=str(DEFAULT_TEST_DATA_PATH))
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--n-estimators", type=int, default=None, help="Default: 100")
    parser.add_argument("--max-depth", type=int, default=None, help="Default: unlimited")
    parser.add_argument(
        "--positive-label",
        default=None,
        help="Positive class for precision/recall/F1 (binary targets). Default: last class in sorted order.",
    )
    parser.add_argument("--drop-duplicates", action="store_true", help="Drop exact duplicate rows.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rf_params = {}
    if args.n_estimators is not None:
        rf_params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        rf_params["max_depth"] = args.max_depth
    run_training(
        data_path=args.data,
        target_column=args.target,
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        rf_params=rf_params,
        positive_label=args.positive_label,
        drop_duplicates=args.drop_duplicates,
    )


if __name__ == "__main__":
    main()
