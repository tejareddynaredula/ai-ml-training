"""
Day 12 - Task 2: Validate Pearson Correlation

Compare our from-scratch Pearson correlation with
NumPy and SciPy implementations.
"""

import numpy as np
from scipy.stats import pearsonr

from task1_pearson_correlation import pearson_correlation


def main():
    """Validate Pearson correlation results."""

    x = np.array([10, 20, 30, 40, 50])
    y = np.array([15, 25, 35, 45, 60])

    # Our from-scratch calculation
    our_result = pearson_correlation(x, y)

    # NumPy calculation
    numpy_result = np.corrcoef(x, y)[0, 1]

    # SciPy calculation
    scipy_result = pearsonr(x, y).statistic

    print("TASK 2: PEARSON CORRELATION VALIDATION")
    print("-" * 55)
    print(f"Our calculation : {our_result:.10f}")
    print(f"NumPy           : {numpy_result:.10f}")
    print(f"SciPy           : {scipy_result:.10f}")

    numpy_match = np.isclose(
        our_result,
        numpy_result,
        atol=1e-6
    )

    scipy_match = np.isclose(
        our_result,
        scipy_result,
        atol=1e-6
    )

    print("-" * 55)
    print(f"Matches NumPy: {numpy_match}")
    print(f"Matches SciPy: {scipy_match}")

    if numpy_match and scipy_match:
        print("\nValidation successful!")
    else:
        print("\nValidation failed!")


if __name__ == "__main__":
    main()