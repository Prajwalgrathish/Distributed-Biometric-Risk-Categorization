"""Saving and loading the trained model with joblib.

The saved file is a dict ("bundle") holding the fitted pipeline plus the
metadata needed for inference (feature columns, target name, positive label).
"""
from pathlib import Path

import joblib

REQUIRED_KEYS = {"pipeline", "target_column", "feature_columns"}


def save_model(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Train one first with `python -m src.train`."
        )
    bundle = joblib.load(path)
    missing = REQUIRED_KEYS - set(bundle)
    if missing:
        raise ValueError(f"Model file is missing expected entries: {sorted(missing)}")
    return bundle
