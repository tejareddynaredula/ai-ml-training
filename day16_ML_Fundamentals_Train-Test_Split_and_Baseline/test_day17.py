import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow importing the Day 17 module from this folder
sys.path.insert(0, str(Path(__file__).parent))

from day17_feature_engineering import (
    add_engineered_features,
    build_preprocessor,
    get_feature_names,
    select_top_features,
)


def sample_data():
    return pd.DataFrame({
        "age": [20, 30, np.nan, 40, 35],
        "city": ["Hyderabad", "Chennai", "Hyderabad", "Delhi", "Chennai"],
    })


def test_preprocessor_removes_missing_values():
    df = sample_data()
    prep = build_preprocessor(["age"], ["city"])
    result = prep.fit_transform(df)

    assert not np.isnan(result).any()


def test_preprocessor_handles_unseen_category():
    train = sample_data()
    prep = build_preprocessor(["age"], ["city"])
    prep.fit(train)

    new_data = pd.DataFrame({
        "age": [25],
        "city": ["Mumbai"],
    })

    result = prep.transform(new_data)
    assert result.shape[0] == 1
    assert not np.isnan(result).any()


def test_engineered_features_do_not_mutate_input():
    df = sample_data()
    original = df.copy(deep=True)

    add_engineered_features(df)

    pd.testing.assert_frame_equal(df, original)


def test_engineered_features_add_at_least_four_columns():
    df = sample_data()
    result = add_engineered_features(df)

    assert result.shape[1] >= df.shape[1] + 4


def test_preprocessor_returns_feature_names():
    df = sample_data()
    prep = build_preprocessor(["age"], ["city"])
    prep.fit(df)

    names = get_feature_names(prep)

    assert all(isinstance(name, str) for name in names)
    assert len(names) == prep.transform(df).shape[1]


def test_preprocessor_save_and_reload(tmp_path):
    df = sample_data()
    prep = build_preprocessor(["age"], ["city"])
    prep.fit(df)

    before = prep.transform(df)
    file_path = tmp_path / "preprocessor.joblib"
    joblib.dump(prep, file_path)

    loaded_prep = joblib.load(file_path)
    after = loaded_prep.transform(df)

    assert np.allclose(before, after)


def test_select_top_features_returns_requested_number():
    X = pd.DataFrame({
        "experience": [1, 2, 3, 4, 5, 6],
        "hours": [10, 20, 30, 40, 50, 60],
        "random_feature": [4, 1, 5, 2, 6, 3],
    })
    y = np.array([10, 20, 30, 40, 50, 60])

    selected = select_top_features(X, y, k=2)

    assert len(selected) == 2
    assert all(name in X.columns for name in selected)


def test_preprocessor_output_has_expected_rows_and_columns():
    df = sample_data()
    prep = build_preprocessor(["age"], ["city"])
    result = prep.fit_transform(df)

    assert result.shape[0] == len(df)
    assert result.shape[1] == 4
