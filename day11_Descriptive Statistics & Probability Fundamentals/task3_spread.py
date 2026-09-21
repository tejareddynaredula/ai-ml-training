
"""
Day 11 - Task 3: Spread Calculations

Calculate variance, standard deviation, range, and IQR
from scratch using NumPy.
"""

import numpy as np


def compute_spread(data: np.ndarray) -> dict:
    """
    Calculate spread measures from scratch.

    Population variance and standard deviation use ddof=0.
    Sample variance and standard deviation use ddof=1.

    ddof=0: Use when data represents the entire population.
    ddof=1: Use when data is a sample of a larger population.
    """

    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        raise ValueError("Data cannot be empty.")

    # Population variance:
    # Sum of squared differences from the mean / number of values
    mean_value = np.sum(data) / len(data)

    variance = np.sum(
        (data - mean_value) ** 2
    ) / len(data)

    # Standard deviation = Square root of variance
    standard_deviation = np.sqrt(variance)

    # Range = Maximum value - Minimum value
    data_range = np.max(data) - np.min(data)

    # IQR = Third quartile - First quartile
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    return {
        "variance": float(variance),
        "std": float(standard_deviation),
        "range": float(data_range),
        "iqr": float(iqr)
    }


def main() -> None:
    """Test spread calculations."""

    data = np.array([2, 4, 4, 6, 8])

    result = compute_spread(data)

    print("TASK 3: SPREAD CALCULATIONS")
    print("-" * 40)
    print(f"Data: {data}")
    print(f"Variance: {result['variance']:.6f}")
    print(f"Standard Deviation: {result['std']:.6f}")
    print(f"Range: {result['range']:.6f}")
    print(f"IQR: {result['iqr']:.6f}")


if __name__ == "__main__":
    main()