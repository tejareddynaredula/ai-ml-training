"""
Day 12 - Task 5: Normality Check

Use the Shapiro-Wilk test to check whether
numeric dataset columns are normally distributed.
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro


def normality_check(data: np.ndarray) -> dict:
    """
    Perform the Shapiro-Wilk normality test.

    Interpretation:
    p-value > 0.05:
        No strong evidence against normality.

    p-value <= 0.05:
        Evidence that the data is not normally distributed.
    """

    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]

    statistic, p_value = shapiro(data)

    if p_value > 0.05:
        interpretation = (
            "No strong evidence against normality."
        )
    else:
        interpretation = (
            "Evidence suggests the data is not normally distributed."
        )

    return {
        "statistic": statistic,
        "p_value": p_value,
        "interpretation": interpretation,
    }


def find_dataset() -> str:
    """Find the Epic 2 cleaned dataset."""

    paths = [
        "day9_Exploratory Data Analysis (EDA) Deliverable/data/cleaned_dataset.csv",
        "day9_Exploratory Data Analysis (EDA) Deliverable\\data\\cleaned_dataset.csv",
    ]

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8"):
                return path
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "cleaned_dataset.csv was not found."
    )


def main():
    """Run normality tests on two numeric columns."""

    dataset_path = find_dataset()
    df = pd.read_csv(dataset_path)

    columns = ["Quantity", "Unit_Price"]

    print("TASK 5: SHAPIRO-WILK NORMALITY TEST")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")

    for column in columns:
        result = normality_check(df[column].to_numpy())

        print(f"\nColumn: {column}")
        print("-" * 60)
        print(f"Shapiro-Wilk statistic: {result['statistic']:.6f}")
        print(f"P-value: {result['p_value']:.6f}")
        print(f"Interpretation: {result['interpretation']}")


if __name__ == "__main__":
    main()