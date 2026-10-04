# Data

| Folder | Contents | In Git? |
|---|---|---|
| `raw/` | Your original CSV dataset (200K+ biometric records) | No (ignored) |
| `processed/` | Held-out test set written by `python -m src.train` | No (ignored) |
| `sample/` | `sample_data.csv` — **synthetic demo data** | Yes |

## Using your dataset
1. Copy your CSV into `data/raw/`.
2. Run `python -m src.train --data data/raw/<file>.csv --target <TARGET_COLUMN>`.

The dataset must be a CSV with one column holding the risk category (the target) and feature columns. Numeric columns are detected automatically; all other columns are treated as categorical.

## Dataset details (fill in)
- Source: `TODO`
- Number of records: 200K+ (as stated by the author)
- Features: `TODO`
- Target column and labels: `TODO`

## About the sample data
`sample/sample_data.csv` is **synthetic**, created by `python -m src.generate_sample_data` (scikit-learn `make_classification`, 2,000 rows, 6 numeric features, 1 categorical column, ~2% missing values). Its columns and labels have no biometric meaning and it must not be presented as the project dataset.
