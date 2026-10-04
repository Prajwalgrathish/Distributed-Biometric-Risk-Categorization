import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    clean_data,
    inspect_data,
    load_data,
    split_features_target,
    split_train_test,
)
from src.feature_engineering import build_preprocessor, identify_column_types


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "num_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "cat_b": ["x", "y", "x", None, "y", "x", "y", "x", "y", "x"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, np.nan],
        }
    )


def test_load_data_reads_csv(tmp_path, small_df):
    path = tmp_path / "data.csv"
    small_df.to_csv(path, index=False)
    loaded = load_data(path)
    assert loaded.shape == small_df.shape


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


def test_inspect_data_reports_missing_values(small_df):
    summary = inspect_data(small_df, "target")
    assert summary["n_rows"] == 10
    assert summary["missing_values"]["num_a"] == 1
    assert summary["missing_values"]["cat_b"] == 1


def test_clean_data_drops_rows_with_missing_target(small_df):
    cleaned = clean_data(small_df, "target")
    assert len(cleaned) == 9
    assert cleaned["target"].notna().all()


def test_clean_data_unknown_target_raises(small_df):
    with pytest.raises(KeyError):
        clean_data(small_df, "not_a_column")


def test_split_features_target(small_df):
    X, y = split_features_target(small_df, "target")
    assert "target" not in X.columns
    assert len(X) == len(y)


def test_split_train_test_is_reproducible_and_sized():
    X = pd.DataFrame({"f": range(100)})
    y = pd.Series([0, 1] * 50)
    first = split_train_test(X, y, test_size=0.2, random_state=7)
    second = split_train_test(X, y, test_size=0.2, random_state=7)
    assert len(first[1]) == 20
    assert first[1].index.tolist() == second[1].index.tolist()
    assert first[3].mean() == pytest.approx(0.5)  # stratified


def test_identify_column_types(small_df):
    X, _ = split_features_target(small_df, "target")
    numeric, categorical = identify_column_types(X)
    assert numeric == ["num_a"]
    assert categorical == ["cat_b"]


def test_preprocessor_imputes_and_encodes(small_df):
    X, _ = split_features_target(small_df, "target")
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape == (10, 2)
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_category(small_df):
    X, _ = split_features_target(small_df, "target")
    preprocessor = build_preprocessor(X).fit(X)
    new = pd.DataFrame({"num_a": [3.0], "cat_b": ["never_seen"]})
    assert not np.isnan(preprocessor.transform(new)).any()
