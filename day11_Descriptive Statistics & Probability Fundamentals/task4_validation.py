
"""
Day 11 - Task 4: Cross-Validation

Compare from-scratch statistics with NumPy, Pandas,
and SciPy library implementations.
"""

import numpy as np
import pandas as pd
from scipy import stats

from task2_central_tendency import compute_central_tendency
from task3_spread import compute_spread


def validate_statistics(data: np.ndarray) -> None:
    """Compare custom calculations with library results."""

    data = np.asarray(data, dtype=float)

    custom_central = compute_central_tendency(data)
    custom_spread = compute_spread(data)

    series = pd.Series(data)

    library_results = {
        "Mean": series.mean(),
        "Median": series.median(),
        "Variance": np.var(data, ddof=0),
        "Standard Deviation": np.std(data, ddof=0),
        "Range": np.ptp(data),
        "IQR": stats.iqr(data),
    }

    custom_results = {
        "Mean": custom_central["mean"],
        "Median": custom_central["median"],
        "Variance": custom_spread["variance"],
        "Standard Deviation": custom_spread["std"],
        "Range": custom_spread["range"],
        "IQR": custom_spread["iqr"],
    }

    print("TASK 4: CROSS-VALIDATION")
    print("=" * 75)
    print(
        f"{'Statistic':<22}"
        f"{'From Scratch':<20}"
        f"{'Library':<20}"
        f"{'Match':<10}"
    )
    print("-" * 75)

    for name in custom_results:
        custom_value = custom_results[name]
        library_value = library_results[name]

        matches = np.isclose(
            custom_value,
            library_value,
            atol=1e-6
        )

        print(
            f"{name:<22}"
            f"{custom_value:<20.6f}"
            f"{library_value:<20.6f}"
            f"{'YES' if matches else 'NO':<10}"
        )

        assert matches, f"{name} does not match."

    print("-" * 75)
    print("All statistics match within 1e-6 tolerance.")


def main() -> None:
    """Run validation with sample data."""

    data = np.array([2, 4, 4, 6, 8])

    validate_statistics(data)


if __name__ == "__main__":
    main()