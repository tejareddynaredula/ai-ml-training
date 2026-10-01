from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    """
    Calculate MAE, MSE, RMSE, and R² from scratch.

    R² is undefined when all actual values are identical.
    In that case, this function returns NaN.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Inputs must contain only finite values.")

    errors = y_true - y_pred

    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    residual_sum_squares = np.sum(errors ** 2)

    if total_sum_squares == 0:
        r2 = float("nan")
    else:
        r2 = 1 - (residual_sum_squares / total_sum_squares)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def plot_residuals(
    y_true,
    y_pred,
    save_path: str,
) -> None:
    """
    Save residuals and predicted-versus-actual plots.

    Interpretation:
    Residuals scattered around zero without a clear pattern suggest
    that the model errors are not strongly systematic. Large patterns
    or curves may indicate that the model is missing relationships.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Inputs must contain only finite values.")

    residuals = y_true - y_pred

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residual plot: points should ideally be scattered around zero.
    axes[0].scatter(y_pred, residuals, alpha=0.7)
    axes[0].axhline(y=0, linestyle="--")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].set_xlabel("Predicted values")
    axes[0].set_ylabel("Residuals (actual - predicted)")
    axes[0].grid(True, alpha=0.3)

    # Predicted vs actual: points close to this line indicate better fit.
    lower = min(np.min(y_true), np.min(y_pred))
    upper = max(np.max(y_true), np.max(y_pred))

    axes[1].scatter(y_true, y_pred, alpha=0.7)
    axes[1].plot([lower, upper], [lower, upper], linestyle="--")
    axes[1].set_title("Predicted vs Actual")
    axes[1].set_xlabel("Actual values")
    axes[1].set_ylabel("Predicted values")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Example predictions for checking the metrics and plots.
    y_true = np.array([3, 5, 2.5, 7, 4.5, 8], dtype=float)
    y_pred = np.array([2.8, 5.2, 2.4, 6.5, 4.7, 7.5], dtype=float)

    metrics = regression_metrics(y_true, y_pred)

    print("Regression Metrics")
    print("------------------")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    chart_path = (
        "day18_Linear_Regression/charts/"
        "residuals_and_predictions.png"
    )
    plot_residuals(y_true, y_pred, chart_path)
    print(f"\nPlots saved to: {chart_path}")
