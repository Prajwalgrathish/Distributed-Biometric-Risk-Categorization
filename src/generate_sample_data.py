"""Generate a small SYNTHETIC dataset for demonstrating the pipeline.

This is NOT the project's biometric dataset. Column names and labels are
generic placeholders; the data has no biometric meaning.

Usage:
    python -m src.generate_sample_data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.config import DEFAULT_RANDOM_STATE, SAMPLE_DATA_PATH, SAMPLE_TARGET_COLUMN


def generate_sample_data(n_samples=2000, random_state=DEFAULT_RANDOM_STATE):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        flip_y=0.02,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"sample_feature_{i}" for i in range(1, 7)])

    rng = np.random.default_rng(random_state)
    df["sample_category"] = rng.choice(["A", "B", "C"], size=n_samples)
    # A few missing values so the imputation path is exercised.
    df.loc[rng.random(n_samples) < 0.02, "sample_feature_2"] = np.nan
    df.loc[rng.random(n_samples) < 0.02, "sample_category"] = np.nan

    df[SAMPLE_TARGET_COLUMN] = y
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic demo data.")
    parser.add_argument("--output", default=str(SAMPLE_DATA_PATH))
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args(argv)

    df = generate_sample_data(args.n_samples)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} synthetic demo rows to {args.output}")


if __name__ == "__main__":
    main()
