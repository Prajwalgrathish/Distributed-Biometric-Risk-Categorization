"""Predict risk categories for new records using a saved model.

Usage:
    python -m src.predict --input new_records.csv --output predictions.csv
    python -m src.predict --json '{"col_a": 1.2, "col_b": "x"}'
"""
import argparse
import json

import pandas as pd

from src.config import DEFAULT_MODEL_PATH
from src.data_preprocessing import load_data
from src.model_io import load_model


def predict_dataframe(bundle, df):
    """Return predictions and class probabilities for each row of df.

    Extra columns (e.g. the target) are ignored; missing feature columns raise an error.
    """
    missing = [col for col in bundle["feature_columns"] if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    X = df[bundle["feature_columns"]]
    pipeline = bundle["pipeline"]
    result = pd.DataFrame({"prediction": pipeline.predict(X)}, index=df.index)
    for cls, column in zip(pipeline.classes_, pipeline.predict_proba(X).T):
        result[f"probability_{cls}"] = column
    return result


def predict_records(bundle, records):
    """Predict from a list of dicts (one dict per record)."""
    return predict_dataframe(bundle, pd.DataFrame(records))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict with a saved Random Forest model.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="CSV file containing the feature columns.")
    source.add_argument("--json", help="A single record as a JSON object.")
    parser.add_argument("--output", help="Write predictions to this CSV (only with --input).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle = load_model(args.model_path)
    if args.json:
        predictions = predict_records(bundle, [json.loads(args.json)])
    else:
        predictions = predict_dataframe(bundle, load_data(args.input))

    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"Wrote {len(predictions):,} predictions to {args.output}")
    else:
        print(predictions.head(20).to_string())


if __name__ == "__main__":
    main()
