import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def gradient_descent_fit(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, list]:
    """
    Fit linear regression using gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix. Include a column of ones if an intercept is needed.
    y : np.ndarray
        Target values.
    lr : float
        Learning rate.
    epochs : int
        Number of training iterations.

    Returns
    -------
    tuple[np.ndarray, list]
        Learned coefficients and the loss history.

    Notes
    -----
    Loss is calculated as mean squared error divided by 2.
    The gradient is X.T @ (predictions - y) / number_of_samples.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a one- or two-dimensional array.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    if X.shape[0] == 0:
        raise ValueError("X and y cannot be empty.")

    if lr <= 0:
        raise ValueError("Learning rate must be greater than zero.")

    if epochs <= 0:
        raise ValueError("Epochs must be greater than zero.")

    number_of_samples = X.shape[0]
    coefficients = np.zeros(X.shape[1], dtype=float)
    loss_history = []

    for _ in range(epochs):
        predictions = X @ coefficients
        errors = predictions - y

        loss = np.mean(errors ** 2) / 2
        loss_history.append(float(loss))

        gradient = (X.T @ errors) / number_of_samples
        coefficients -= lr * gradient

    # Record the final loss after the last coefficient update.
    final_errors = X @ coefficients - y
    final_loss = np.mean(final_errors ** 2) / 2
    loss_history.append(float(final_loss))

    return coefficients, loss_history


def plot_loss_curve(
    loss_history: list,
    save_path: str,
) -> None:
    """Plot and save the loss across gradient descent iterations."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss_history)), loss_history)
    plt.title("Gradient Descent Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE / 2)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    # Demonstration using a simple linear relationship.
    x = np.arange(0, 11, dtype=float)
    y = 3 * x + 2

    # Add a column of ones so the model learns an intercept.
    X = np.column_stack((np.ones(len(x)), x))

    learning_rate = 0.01
    epochs = 3000

    coefficients, loss_history = gradient_descent_fit(
        X, y, lr=learning_rate, epochs=epochs
    )

    chart_path = "day18_Linear_Regression/charts/loss_curve.png"
    plot_loss_curve(loss_history, chart_path)

    print("Gradient Descent Results")
    print("------------------------")
    print("Expected coefficients: [2, 3]")
    print("Learned coefficients: ", coefficients)
    print(f"Initial loss: {loss_history[0]:.6f}")
    print(f"Final loss:   {loss_history[-1]:.6f}")
    print(f"Loss curve saved to: {chart_path}")

    # The loss should decrease for this dataset and learning rate.
    assert loss_history[-1] < loss_history[0]
