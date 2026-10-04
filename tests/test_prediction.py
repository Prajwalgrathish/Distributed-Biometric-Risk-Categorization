import pytest

from src.config import SAMPLE_TARGET_COLUMN
from src.data_preprocessing import split_features_target, split_train_test
from src.evaluate import evaluate_model
from src.generate_sample_data import generate_sample_data
from src.model_io import load_model, save_model
from src.predict import predict_dataframe, predict_records
from src.train import resolve_positive_label, run_training, train_model

SMALL_RF = {"n_estimators": 10}


@pytest.fixture(scope="module")
def trained():
    df = generate_sample_data(n_samples=400).dropna(subset=[SAMPLE_TARGET_COLUMN])
    X, y = split_features_target(df, SAMPLE_TARGET_COLUMN)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    pipeline = train_model(X_train, y_train, rf_params=SMALL_RF)
    bundle = {
        "pipeline": pipeline,
        "target_column": SAMPLE_TARGET_COLUMN,
        "feature_columns": X.columns.tolist(),
        "positive_label": resolve_positive_label(pipeline.classes_.tolist()),
    }
    return bundle, X_test, y_test


def test_save_and_load_roundtrip_gives_same_predictions(trained, tmp_path):
    bundle, X_test, _ = trained
    path = save_model(bundle, tmp_path / "model.joblib")
    loaded = load_model(path)
    original = predict_dataframe(bundle, X_test)
    reloaded = predict_dataframe(loaded, X_test)
    assert original.equals(reloaded)


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")


def test_predict_dataframe_output_shape_and_probabilities(trained):
    bundle, X_test, _ = trained
    result = predict_dataframe(bundle, X_test)
    assert len(result) == len(X_test)
    assert "prediction" in result.columns
    probability_cols = [c for c in result.columns if c.startswith("probability_")]
    assert len(probability_cols) == 2
    assert result[probability_cols].sum(axis=1).round(6).eq(1.0).all()


def test_predict_ignores_extra_columns(trained):
    bundle, X_test, y_test = trained
    with_target = X_test.assign(**{SAMPLE_TARGET_COLUMN: y_test})
    assert len(predict_dataframe(bundle, with_target)) == len(X_test)


def test_predict_missing_feature_column_raises(trained):
    bundle, X_test, _ = trained
    with pytest.raises(ValueError):
        predict_dataframe(bundle, X_test.drop(columns=["sample_feature_1"]))


def test_predict_records_single_dict(trained):
    bundle, X_test, _ = trained
    record = X_test.iloc[0].to_dict()
    assert len(predict_records(bundle, [record])) == 1


def test_evaluate_model_returns_valid_metrics(trained):
    bundle, X_test, y_test = trained
    metrics = evaluate_model(bundle, X_test, y_test)
    for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0.0 <= metrics[name] <= 1.0
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == len(y_test)


def test_resolve_positive_label():
    assert resolve_positive_label([0, 1]) == 1
    assert resolve_positive_label([0, 1], requested="0") == 0
    assert resolve_positive_label([0, 1, 2]) is None
    with pytest.raises(ValueError):
        resolve_positive_label([0, 1], requested="9")


def test_run_training_end_to_end_saves_model_and_test_set(tmp_path):
    csv_path = tmp_path / "data.csv"
    generate_sample_data(n_samples=300).to_csv(csv_path, index=False)
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    run_training(csv_path, SAMPLE_TARGET_COLUMN, model_path, test_path, rf_params=SMALL_RF)
    assert model_path.exists() and test_path.exists()
    assert load_model(model_path)["target_column"] == SAMPLE_TARGET_COLUMN
