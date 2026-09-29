from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score


DATA_DIR = Path(__file__).parent / "data"


def load_ml_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a saved regression or classification dataset."""
    files = {
        "diabetes": "diabetes_regression.csv",
        "breast_cancer": "breast_cancer_classification.csv",
    }

    if name not in files:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(files)}")

    df = pd.read_csv(DATA_DIR / files[name])

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    stratify: bool = False,
) -> tuple:
    """Split data into training and testing sets with validation."""
    if len(X) == 0 or len(y) == 0:
        raise ValueError("X and y must not be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    stratify_values = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_values,
    )


def get_baseline_scores(
    X_train,
    X_test,
    y_train,
    y_test,
    task: str,
) -> dict:
    """Train a simple baseline model and return its test scores."""
    if task == "regression":
        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        return {
            "model": "DummyRegressor (mean)",
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": rmse,
            "R2": r2_score(y_test, predictions),
        }

    if task == "classification":
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        return {
            "model": "DummyClassifier (most_frequent)",
            "accuracy": accuracy_score(y_test, predictions),
        }

    raise ValueError("task must be 'regression' or 'classification'.")


def demonstrate_leakage(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compare scaling when the scaler sees all data versus training data only.

    Fitting a scaler on all data lets information from the test set influence
    preprocessing. The correct approach is to fit it on training data only.
    """
    X_train, X_test, _, _ = split_data(X, y)

    # Incorrect: scaler learns from both training and test data.
    full_data_scaler = StandardScaler()
    full_data_scaler.fit(X)
    test_scaled_with_leakage = full_data_scaler.transform(X_test)

    # Correct: scaler learns only from the training data.
    train_only_scaler = StandardScaler()
    train_only_scaler.fit(X_train)
    test_scaled_correctly = train_only_scaler.transform(X_test)

    max_difference = float(
        np.max(np.abs(test_scaled_with_leakage - test_scaled_correctly))
    )

    return {
        "incorrect_method": "Fit StandardScaler on the full dataset",
        "correct_method": "Fit StandardScaler on training data only",
        "max_test_value_difference": max_difference,
        "difference_measurable": max_difference > 1e-10,
    }


class MLExperiment:
    """Store the key details and baseline result of an ML experiment."""

    def __init__(
        self,
        dataset_name: str,
        train_size: int,
        test_size: int,
        baseline_score: dict,
    ):
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.test_size = test_size
        self.baseline_score = baseline_score

    def summary(self) -> str:
        return (
            f"Dataset: {self.dataset_name}\n"
            f"Training rows: {self.train_size}\n"
            f"Testing rows: {self.test_size}\n"
            f"Baseline score: {self.baseline_score}"
        )


def run_experiment(name: str, task: str, stratify: bool = False) -> None:
    """Load data, split it, evaluate a baseline, and print a summary."""
    X, y = load_ml_dataset(name)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"Rows: {len(X)} | Features: {X.shape[1]}")

    if task == "classification":
        print("\nClass distribution before splitting:")
        print(y.value_counts(normalize=True).sort_index().round(4))

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, stratify=stratify
    )

    if task == "classification":
        print("\nClass distribution in training set:")
        print(y_train.value_counts(normalize=True).sort_index().round(4))
        print("\nClass distribution in testing set:")
        print(y_test.value_counts(normalize=True).sort_index().round(4))

    scores = get_baseline_scores(
        X_train, X_test, y_train, y_test, task
    )

    experiment = MLExperiment(
        dataset_name=name,
        train_size=len(X_train),
        test_size=len(X_test),
        baseline_score=scores,
    )

    print("\nExperiment summary:")
    print(experiment.summary())


def main() -> None:
    # Regression experiment
    run_experiment("diabetes", task="regression")

    # Classification experiment; stratification preserves class proportions.
    run_experiment(
        "breast_cancer",
        task="classification",
        stratify=True,
    )

    # Demonstrate preprocessing leakage on the regression dataset.
    X, y = load_ml_dataset("diabetes")
    print(f"\n{'=' * 60}")
    print("Data leakage demonstration:")
    for key, value in demonstrate_leakage(X, y).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


# Learning notes:
# Overfitting: The model learns the training data too closely, including noise,
# so it may perform poorly on new data.
#
# Underfitting: The model is too simple to learn important patterns in the data.
#
# Bias: Error caused by assumptions that make a model too simple.
# Variance: How much the model's predictions change when its training data changes.
# A good model aims to balance bias and variance.
