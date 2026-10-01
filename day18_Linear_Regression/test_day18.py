import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR / "day18_Linear_Regression"))
sys.path.insert(0, str(PROJECT_DIR / "day17_Feature_Engineering"))

from task2_normal_equation import normal_equation_fit
from task3_gradient_descent import gradient_descent_fit, plot_loss_curve
from task4_regression_metrics_and_plots import (
    regression_metrics,
    plot_residuals,
)
from task5_sklearn_models import (
    load_data,
    build_model_pipeline,
    evaluate_model,
    get_model_coefficients,
)


# --------------------------------------------------
# Task 2: Normal Equation
# --------------------------------------------------

def test_normal_equation_finds_correct_coefficients():
    x = np.arange(1, 6, dtype=float)
    X = np.column_stack((np.ones(len(x)), x))
    y = 2 + 3 * x

    coefficients = normal_equation_fit(X, y)

    np.testing.assert_allclose(
        coefficients, [2, 3], atol=1e-8
    )


def test_normal_equation_handles_singular_matrix():
    x = np.arange(1, 6, dtype=float)
    X = np.column_stack((x, x))
    y = 4 * x

    coefficients = normal_equation_fit(X, y)

    predictions = X @ coefficients
    np.testing.assert_allclose(predictions, y, atol=1e-8)


def test_normal_equation_rejects_mismatched_lengths():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2])

    with pytest.raises(ValueError):
        normal_equation_fit(X, y)


# --------------------------------------------------
# Task 3: Gradient Descent
# --------------------------------------------------

def test_gradient_descent_learns_linear_relationship():
    x = np.arange(0, 11, dtype=float)
    X = np.column_stack((np.ones(len(x)), x))
    y = 2 + 3 * x

    coefficients, loss_history = gradient_descent_fit(
        X, y, lr=0.01, epochs=3000
    )

    np.testing.assert_allclose(
        coefficients, [2, 3], atol=0.01
    )
    assert loss_history[-1] < loss_history[0]


def test_gradient_descent_rejects_zero_learning_rate():
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])

    with pytest.raises(ValueError):
        gradient_descent_fit(X, y, lr=0, epochs=100)


def test_loss_curve_is_saved(tmp_path):
    loss_history = [10.0, 5.0, 2.0, 0.5]
    output_file = tmp_path / "charts" / "loss.png"

    plot_loss_curve(loss_history, str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0


# --------------------------------------------------
# Task 4: Regression Metrics and Plots
# --------------------------------------------------

def test_regression_metrics_calculate_expected_values():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["MAE"] == pytest.approx(1 / 3)
    assert metrics["MSE"] == pytest.approx(1 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["R2"] == pytest.approx(0.5)


def test_regression_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        regression_metrics([1, 2, 3], [1, 2])


def test_regression_metrics_returns_nan_r2_for_constant_target():
    metrics = regression_metrics([5, 5, 5], [4, 5, 6])

    assert np.isnan(metrics["R2"])


def test_residual_plot_is_saved(tmp_path):
    y_true = np.array([3, 5, 2.5, 7, 4.5, 8])
    y_pred = np.array([2.8, 5.2, 2.4, 6.5, 4.7, 7.5])
    output_file = tmp_path / "plots" / "residuals.png"

    plot_residuals(y_true, y_pred, str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0


# --------------------------------------------------
# Task 5: Scikit-learn Models
# --------------------------------------------------

def test_load_data_returns_expected_features():
    X, y = load_data()

    assert len(X) == len(y)
    assert len(X) > 0
    assert "target" not in X.columns
    assert X.shape[1] == 10
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)


def test_sklearn_pipeline_evaluates_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_model_pipeline(LinearRegression())

    result = evaluate_model(
        "Linear Regression",
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    assert result["Model"] == "Linear Regression"
    assert np.isfinite(result["MAE"])
    assert np.isfinite(result["RMSE"])
    assert np.isfinite(result["R²"])


def test_fitted_pipeline_returns_ten_coefficients():
    X, y = load_data()

    pipeline = build_model_pipeline(LinearRegression())
    pipeline.fit(X, y)

    coefficients = get_model_coefficients(pipeline)

    assert len(coefficients) == 10
    assert np.all(np.isfinite(coefficients.to_numpy()))
