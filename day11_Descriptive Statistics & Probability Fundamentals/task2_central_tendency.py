
"""
Day 11 - Task 2: Central Tendency

Calculate mean, median, and mode from scratch using NumPy.
"""

import numpy as np
from collections import Counter


def compute_central_tendency(data: np.ndarray) -> dict:
    """Calculate mean, median, and mode."""

    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        raise ValueError("Data cannot be empty.")

    # Mean = Sum of values / Number of values
    mean_value = np.sum(data) / len(data)

    # Median
    sorted_data = np.sort(data)
    n = len(sorted_data)

    if n % 2 == 0:
        median_value = (
            sorted_data[n // 2 - 1]
            + sorted_data[n // 2]
        ) / 2
    else:
        median_value = sorted_data[n // 2]

    # Mode
    unique_values, counts = np.unique(
        data,
        return_counts=True
    )

    max_count = np.max(counts)
    mode_values = unique_values[counts == max_count]

    return {
        "mean": float(mean_value),
        "median": float(median_value),
        "mode": mode_values.tolist()
    }


def main() -> None:
    """Test central tendency calculations."""

    data = np.array([2, 4, 4, 6, 8])

    result = compute_central_tendency(data)

    print("TASK 2: CENTRAL TENDENCY")
    print("-" * 40)
    print(f"Data: {data}")
    print(f"Mean: {result['mean']}")
    print(f"Median: {result['median']}")
    print(f"Mode: {result['mode']}")


if __name__ == "__main__":
    main()