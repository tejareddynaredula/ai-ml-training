"""
Day 18 - Task 5: Scikit-learn Regression Models

Models:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression

Uses the preprocessing pipeline created in Day 17.
Evaluates models using MAE, RMSE, and R².
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Paths and Day 17 preprocessor
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "day16_ML_Fundamentals_Train-Test_Split_and_Baseline"
    / "data"
    / "diabetes_regression.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import the preprocessing function from Day 17.
DAY17_DIR = PROJECT_DIR / "day17_Feature_Engineering"
sys.path.insert(0, str(DAY17_DIR))

from day17_feature_engineering import build_preprocessor


# --------------------------------------------------
# Load and prepare the dataset
# --------------------------------------------------

def load_data():
    """Load the diabetes regression dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    X = df.drop(columns=["target"])
    y = df["target"]

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X):
        raise TypeError("All input features must be numeric.")

    return X, y


# --------------------------------------------------
# Build a model pipeline
# --------------------------------------------------

def build_model_pipeline(model):
    """
    Create a pipeline using the Day 17 preprocessor.

    The preprocessor is fitted on training data only
    when the pipeline is trained.
    """

    numeric_cols = [
        "age", "sex", "bmi", "bp", "s1",
        "s2", "s3", "s4", "s5", "s6",
    ]

    categorical_cols = []

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# --------------------------------------------------
# Evaluate models
# --------------------------------------------------

def evaluate_model(model_name, pipeline, X_train, X_test, y_train, y_test):
    """Train a model and calculate regression metrics."""

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
    }


# --------------------------------------------------
# Compare model coefficients
# --------------------------------------------------

def get_model_coefficients(pipeline):
    """Return feature names and fitted model coefficients."""

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    return pd.Series(
        model.coef_,
        index=feature_names,
        name=type(model).__name__,
    )


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=10000),
    }

    pipelines = {}
    results = []

    for model_name, model in models.items():
        pipeline = build_model_pipeline(model)

        result = evaluate_model(
            model_name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        pipelines[model_name] = pipeline
        results.append(result)

    # Model performance comparison
    results_df = pd.DataFrame(results)

    print("\nMODEL PERFORMANCE COMPARISON")
    print(results_df.to_string(index=False, float_format="%.4f"))

    results_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    # Compare coefficients from all three models
    coefficients = pd.concat(
        [
            get_model_coefficients(pipeline)
            for pipeline in pipelines.values()
        ],
        axis=1,
    )

    coefficients.columns = list(pipelines.keys())

    print("\nCOEFFICIENT COMPARISON")
    print(coefficients.to_string(float_format="%.4f"))

    coefficients_path = OUTPUT_DIR / "coefficient_comparison.csv"
    coefficients.to_csv(coefficients_path)

    print("\nFiles saved:")
    print(f"Model comparison: {results_path}")
    print(f"Coefficient comparison: {coefficients_path}")


if __name__ == "__main__":
    main()
