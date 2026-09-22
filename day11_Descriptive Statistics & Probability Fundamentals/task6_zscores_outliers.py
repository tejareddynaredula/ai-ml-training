"""
Day 12 - Task 6: Z-Scores and Outlier Detection

Calculate z-scores and identify values beyond
+/-2 and +/-3 standard deviations.
"""

import numpy as np
import pandas as pd


def compute_zscores(data: np.ndarray) -> np.ndarray:
    """
    Calculate z-scores from scratch.

    Formula:
        z = (x - mean) / standard_deviation
    """

    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        raise ValueError("Data cannot be empty.")

    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        raise ValueError(
            "Z-scores cannot be calculated when standard deviation is zero."
        )

    return (data - mean) / std


def identify_outliers(
    data: np.ndarray,
    threshold: float = 2.0
) -> np.ndarray:
    """Return values whose absolute z-score exceeds the threshold."""

    zscores = compute_zscores(data)

    return data[np.abs(zscores) > threshold]


def main():
    """Calculate z-scores and identify outliers."""

    # Synthetic data with an obvious high-value outlier
    data = np.array([
        10, 11, 12, 12, 13,
        13, 14, 14, 15, 15,
        16, 17, 18, 19, 50
    ], dtype=float)

    zscores = compute_zscores(data)

    outliers_2 = identify_outliers(data, threshold=2)
    outliers_3 = identify_outliers(data, threshold=3)

    print("TASK 6: Z-SCORES AND OUTLIER DETECTION")
    print("=" * 60)

    print(f"Data: {data}")
    print(f"\nMean: {np.mean(data):.4f}")
    print(f"Standard deviation: {np.std(data):.4f}")

    print("\nZ-scores:")
    for value, zscore in zip(data, zscores):
        print(f"Value: {value:6.1f}  Z-score: {zscore:8.4f}")

    print("\nOutliers beyond +/-2 standard deviations:")
    print(outliers_2)

    print("\nOutliers beyond +/-3 standard deviations:")
    print(outliers_3)

    print("\nInterpretation:")
    print(
        "Values beyond +/-2 standard deviations are "
        "potentially unusual."
    )
    print(
        "Values beyond +/-3 standard deviations are "
        "stronger outlier candidates."
    )


if __name__ == "__main__":
    main()