"""
Day 12 - Task 1: Pearson Correlation From Scratch
"""

import numpy as np


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient from scratch."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if len(x) < 2:
        raise ValueError("At least two values are required.")

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum(
        (x - x_mean) * (y - y_mean)
    )

    denominator = np.sqrt(
        np.sum((x - x_mean) ** 2)
        * np.sum((y - y_mean) ** 2)
    )

    if denominator == 0:
        raise ValueError(
            "Correlation cannot be calculated when there is no variation."
        )

    return numerator / denominator


def main():
    """Test Pearson correlation."""

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    result = pearson_correlation(x, y)

    print("TASK 1: PEARSON CORRELATION")
    print("-" * 50)
    print(f"X values: {x}")
    print(f"Y values: {y}")
    print(f"Pearson correlation: {result:.4f}")


if __name__ == "__main__":
    main()