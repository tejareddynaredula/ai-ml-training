"""
Day 17 - Task 2: Numeric Scaling

StandardScaler:
    Centers features around mean 0 and scales them to
    standard deviation 1. Useful for models sensitive
    to feature scales.

MinMaxScaler:
    Scales features to a fixed range, usually 0 to 1.
    Useful when a bounded range is desired. Sensitive
    to outliers.
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression


def scale_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "standard",
):
    """
    Scale numeric features using a fitted training-data scaler.

    Parameters
    ----------
    X_train : pd.DataFrame
        Numeric training features.
    X_test : pd.DataFrame
        Numeric test features.
    scaler_type : str
        "standard" for StandardScaler or "minmax" for MinMaxScaler.

    Returns
    -------
    tuple
        Scaled training data, scaled test data, and fitted scaler.

    Important:
        Fit the scaler on training data only to prevent data leakage.
    """

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(
            "scaler_type must be 'standard' or 'minmax'"
        )

    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler

# Day 17 - Task 3: Categorical Encoding

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    nominal_cols: list,
    ordinal_cols: list,
    ordinal_categories: list | None = None,
):
    """
    Encode categorical features.

    Nominal columns use OneHotEncoder.
    Unknown nominal categories are ignored.

    Ordinal columns use OrdinalEncoder.
    Unknown ordinal categories are encoded as -1.
    """

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    encoders = {}

    if nominal_cols:
        onehot = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        train_values = onehot.fit_transform(
            X_train[nominal_cols]
        )
        test_values = onehot.transform(
            X_test[nominal_cols]
        )

        encoded_cols = onehot.get_feature_names_out(
            nominal_cols
        )

        train_encoded = pd.DataFrame(
            train_values,
            columns=encoded_cols,
            index=X_train.index,
        )

        test_encoded = pd.DataFrame(
            test_values,
            columns=encoded_cols,
            index=X_test.index,
        )

        X_train_encoded = X_train_encoded.drop(
            columns=nominal_cols
        )
        X_test_encoded = X_test_encoded.drop(
            columns=nominal_cols
        )

        X_train_encoded = pd.concat(
            [X_train_encoded, train_encoded], axis=1
        )
        X_test_encoded = pd.concat(
            [X_test_encoded, test_encoded], axis=1
        )

        encoders["onehot"] = onehot

    if ordinal_cols:
        if ordinal_categories is None:
            raise ValueError(
                "Provide ordinal_categories for ordinal columns."
            )

        ordinal = OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        train_values = ordinal.fit_transform(
            X_train[ordinal_cols]
        )
        test_values = ordinal.transform(
            X_test[ordinal_cols]
        )

        # Replace text columns with numeric encoded columns
        X_train_encoded = X_train_encoded.drop(columns=ordinal_cols)
        X_test_encoded = X_test_encoded.drop(columns=ordinal_cols)

        train_ordinal = pd.DataFrame(
            train_values,
            columns=ordinal_cols,
            index=X_train.index,
        )
        test_ordinal = pd.DataFrame(
            test_values,
            columns=ordinal_cols,
            index=X_test.index,
        )

        X_train_encoded = pd.concat(
            [X_train_encoded, train_ordinal], axis=1
        )
        X_test_encoded = pd.concat(
            [X_test_encoded, test_ordinal], axis=1
        )

        encoders["ordinal"] = ordinal

    return X_train_encoded, X_test_encoded, encoders

# Day 17 - Task 4: Handle Missing Values

import numpy as np
from sklearn.impute import SimpleImputer


def impute_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
):
    """
    Fill missing values in numeric and categorical columns.

    Numeric columns:
        Use the median.

    Categorical columns:
        Use the most frequent value.

    Fit imputers on training data only.
    Return the transformed data and fitted imputers.
    """

    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    imputers = {}

    # Impute numeric columns using the median
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="median")

        train_values = numeric_imputer.fit_transform(
            X_train[numeric_cols]
        )
        test_values = numeric_imputer.transform(
            X_test[numeric_cols]
        )

        train_numeric = pd.DataFrame(
            train_values,
            columns=numeric_cols,
            index=X_train.index,
        )

        test_numeric = pd.DataFrame(
            test_values,
            columns=numeric_cols,
            index=X_test.index,
        )

        X_train_imputed = X_train_imputed.drop(
            columns=numeric_cols
        )
        X_test_imputed = X_test_imputed.drop(
            columns=numeric_cols
        )

        X_train_imputed = pd.concat(
            [X_train_imputed, train_numeric], axis=1
        )
        X_test_imputed = pd.concat(
            [X_test_imputed, test_numeric], axis=1
        )

        imputers["numeric"] = numeric_imputer

    # Impute categorical columns using the most frequent value
    if categorical_cols:
        categorical_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        # Convert None values to NaN so the imputer recognizes them
        train_categorical_input = (
            X_train[categorical_cols]
            .astype(object)
            .replace({None: np.nan})
        )
        test_categorical_input = (
            X_test[categorical_cols]
            .astype(object)
            .replace({None: np.nan})
        )

        train_values = categorical_imputer.fit_transform(
            train_categorical_input
        )
        test_values = categorical_imputer.transform(
            test_categorical_input
        )

        train_categorical = pd.DataFrame(
            train_values,
            columns=categorical_cols,
            index=X_train.index,
        )

        test_categorical = pd.DataFrame(
            test_values,
            columns=categorical_cols,
            index=X_test.index,
        )

        X_train_imputed = X_train_imputed.drop(
            columns=categorical_cols
        )
        X_test_imputed = X_test_imputed.drop(
            columns=categorical_cols
        )

        X_train_imputed = pd.concat(
            [X_train_imputed, train_categorical], axis=1
        )
        X_test_imputed = pd.concat(
            [X_test_imputed, test_categorical], axis=1
        )

        imputers["categorical"] = categorical_imputer

    # Keep the original column order
    X_train_imputed = X_train_imputed[X_train.columns]
    X_test_imputed = X_test_imputed[X_test.columns]

    return X_train_imputed, X_test_imputed, imputers

# Day 17 - Task 5: Feature Engineering

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create four new features from the first numeric column.

    The original DataFrame is not modified.
    """

    result = df.copy()

    numeric_cols = result.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_cols:
        raise ValueError(
            "The DataFrame must contain at least one numeric column."
        )

    source_col = numeric_cols[0]
    values = result[source_col]

    result[f"{source_col}_squared"] = values ** 2
    result[f"{source_col}_absolute"] = values.abs()
    result[f"{source_col}_log1p_abs"] = np.log1p(
        values.abs()
    )
    result[f"{source_col}_is_zero"] = (
        values == 0
    ).astype(int)

    return result

# Day 17 - Task 6: Build a preprocessing pipeline

def build_preprocessor(
    numeric_cols: list,
    categorical_cols: list,
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for numeric and categorical columns.

    Numeric:
        Median imputation -> StandardScaler

    Categorical:
        Most-frequent imputation -> OneHotEncoder
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    return preprocessor

# Day 17 - Task 7: Feature Selection

def select_top_features(X, y, k: int) -> list:
    """
    Select the top k features for a numerical target.

    X: DataFrame containing input features.
    y: Numerical target.
    k: Number of features to select.

    Returns a list of selected feature names.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if k < 1 or k > X.shape[1]:
        raise ValueError(
            f"k must be between 1 and {X.shape[1]}."
        )

    selector = SelectKBest(
        score_func=f_regression,
        k=k,
    )

    selector.fit(X, y)

    selected_features = X.columns[
        selector.get_support()
    ].tolist()

    return selected_features

# Day 17 - Task 8: Get Feature Names

def get_feature_names(preprocessor) -> list:
    """
    Return the output feature names from a fitted preprocessor.
    """
    return preprocessor.get_feature_names_out().tolist()
