"""
Day 19 - Logistic Regression from Scratch
"""

from pathlib import Path

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Calculate sigmoid probabilities without overflow warnings."""
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)

    # Separate positive and negative values for numerical stability.
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)

    return result


def logistic_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, list]:
    """
    Train binary logistic regression using gradient descent.

    X should include a column of ones if an intercept is required.
    Returns learned coefficients and the log-loss history.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be one- or two-dimensional.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X cannot be empty.")

    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Inputs must contain only finite values.")

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 and 1.")

    if lr <= 0:
        raise ValueError("Learning rate must be greater than zero.")

    if epochs <= 0:
        raise ValueError("Epochs must be greater than zero.")

    n_samples = X.shape[0]
    coefficients = np.zeros(X.shape[1], dtype=float)
    loss_history = []

    epsilon = 1e-15

    for _ in range(epochs):
        probabilities = sigmoid(X @ coefficients)
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

        loss = -np.mean(
            y * np.log(probabilities)
            + (1 - y) * np.log(1 - probabilities)
        )
        loss_history.append(float(loss))

        gradient = X.T @ (probabilities - y) / n_samples
        coefficients -= lr * gradient

    # Record loss after the final update.
    probabilities = np.clip(
        sigmoid(X @ coefficients), epsilon, 1 - epsilon
    )
    final_loss = -np.mean(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )
    loss_history.append(float(final_loss))

    return coefficients, loss_history
