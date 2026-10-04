"""Run the full pipeline: train -> evaluate -> example prediction.

    python main.py --data data/raw/your_data.csv --target YOUR_TARGET_COLUMN
    python main.py --demo        # synthetic demo data, outputs kept separate
"""
import argparse

from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_TEST_DATA_PATH,
    DEMO_MODEL_PATH,
    DEMO_REPORT_PATH,
    DEMO_TEST_DATA_PATH,
    SAMPLE_DATA_PATH,
    SAMPLE_TARGET_COLUMN,
)
from src.data_preprocessing import load_data, split_features_target
from src.evaluate import run_evaluation
from src.generate_sample_data import generate_sample_data
from src.model_io import load_model
from src.predict import predict_dataframe
from src.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Distributed Biometric Risk Categorization pipeline")
    parser.add_argument("--data", help="Path to your CSV dataset.")
    parser.add_argument("--target", help="Name of the target column in your CSV.")
    parser.add_argument("--positive-label", default=None, help="Positive class for binary targets.")
    parser.add_argument("--demo", action="store_true", help="Use the synthetic demo dataset.")
    args = parser.parse_args()

    if args.demo:
        if not SAMPLE_DATA_PATH.exists():
            SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            generate_sample_data().to_csv(SAMPLE_DATA_PATH, index=False)
        data_path, target = SAMPLE_DATA_PATH, SAMPLE_TARGET_COLUMN
        model_path, test_path, report_path = DEMO_MODEL_PATH, DEMO_TEST_DATA_PATH, DEMO_REPORT_PATH
        label = "SYNTHETIC DEMO DATA - for pipeline demonstration only, not the project dataset"
    elif args.data and args.target:
        data_path, target = args.data, args.target
        model_path, test_path, report_path = DEFAULT_MODEL_PATH, DEFAULT_TEST_DATA_PATH, DEFAULT_REPORT_PATH
        label = "User-supplied dataset"
    else:
        parser.error("Provide --data and --target, or use --demo.")

    print("== 1/3 Training ==")
    run_training(data_path, target, model_path, test_path, positive_label=args.positive_label)

    print("\n== 2/3 Evaluation ==")
    run_evaluation(model_path, test_path, report_path, dataset_label=label)

    print("\n== 3/3 Example prediction (first 5 held-out rows) ==")
    bundle = load_model(model_path)
    X_test, _ = split_features_target(load_data(test_path), bundle["target_column"])
    print(predict_dataframe(bundle, X_test.head(5)).to_string())


if __name__ == "__main__":
    main()
