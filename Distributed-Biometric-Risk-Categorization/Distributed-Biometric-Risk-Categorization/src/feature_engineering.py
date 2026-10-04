"""Feature preparation: column typing, imputation and encoding.

Column types are inferred from the DataFrame dtypes, so nothing here depends on
specific column names:
  - numeric columns     -> median imputation
  - non-numeric columns -> most-frequent imputation + ordinal encoding
Scaling is not applied because Random Forests do not need it.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import DEFAULT_RANDOM_STATE, DEFAULT_RF_PARAMS


def identify_column_types(X):
    """Split feature columns into numeric and non-numeric (categorical) lists."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [col for col in X.columns if col not in numeric]
    return numeric, categorical


def build_preprocessor(X):
    """Build a ColumnTransformer suited to the columns present in X."""
    numeric, categorical = identify_column_types(X)
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        categorical_steps = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        transformers.append(("categorical", categorical_steps, categorical))
    if not transformers:
        raise ValueError("No feature columns found to preprocess.")
    return ColumnTransformer(transformers)


def build_model_pipeline(X, rf_params=None, random_state=DEFAULT_RANDOM_STATE):
    """Preprocessing + Random Forest as one scikit-learn Pipeline."""
    params = {**DEFAULT_RF_PARAMS, **(rf_params or {})}
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X)),
            ("classifier", RandomForestClassifier(random_state=random_state, **params)),
        ]
    )
