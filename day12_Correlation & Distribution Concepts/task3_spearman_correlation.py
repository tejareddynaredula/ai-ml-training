"""
Day 12 - Task 3: Spearman Rank Correlation

Calculate Spearman rank correlation using SciPy.
"""

import numpy as np
from scipy.stats import spearmanr


def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Calculate Spearman rank correlation using SciPy.

    Spearman is useful when:
    - The relationship is monotonic rather than strictly linear.
    - Data contains outliers that may affect Pearson correlation.
    - The data is ordinal/rank-based.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if len(x) < 2:
        raise ValueError("At least two values are required.")

    result = spearmanr(x, y)

    return result.statistic


def main():
    """Run a Spearman correlation example."""

    x = np.array([10, 20, 30, 40, 50])
    y = np.array([15, 25, 35, 45, 60])

    result = spearman_correlation(x, y)

    print("TASK 3: SPEARMAN RANK CORRELATION")
    print("-" * 55)
    print(f"X values: {x}")
    print(f"Y values: {y}")
    print(f"Spearman correlation: {result:.4f}")

    if result > 0:
        print("Interpretation: Positive monotonic relationship.")
    elif result < 0:
        print("Interpretation: Negative monotonic relationship.")
    else:
        print("Interpretation: No monotonic relationship.")


if __name__ == "__main__":
    main()