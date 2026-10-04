"""Shared defaults for the pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation_results.md"

# Demo-only locations. Kept separate so demo output never overwrites real results.
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "sample_data.csv"
SAMPLE_TARGET_COLUMN = "risk_category"
DEMO_MODEL_PATH = PROJECT_ROOT / "models" / "demo_random_forest.joblib"
DEMO_TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demo_test.csv"
DEMO_REPORT_PATH = PROJECT_ROOT / "reports" / "demo_evaluation_results.md"

# Random Forest settings. These are scikit-learn's defaults for the two main
# parameters; no hyperparameter tuning is performed in this project.
DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "n_jobs": -1,
}

# Results documented for the original project (author-supplied, 200K+ records).
# Used ONLY for side-by-side comparison in the report. They are never presented
# as results of the current run.
DOCUMENTED_RESULTS = {
    "accuracy": 0.9931,
    "recall": 0.9979,
    "roc_auc": 0.9997,
}
