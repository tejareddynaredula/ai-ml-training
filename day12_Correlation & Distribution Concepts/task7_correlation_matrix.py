"""
Day 12 - Task 7: Correlation Matrix Report

Generate Pearson and Spearman correlation matrices
for numeric columns in the Epic 2 dataset.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def correlation_matrix_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a full pairwise Pearson + Spearman correlation report.

    The report contains:
    - Pearson correlation
    - Spearman correlation
    """

    numeric_df = df.select_dtypes(include=np.number)

    pearson = numeric_df.corr(method="pearson")
    spearman = numeric_df.corr(method="spearman")

    report = pd.DataFrame(
        index=numeric_df.columns,
        columns=numeric_df.columns,
        dtype=object
    )

    for row in numeric_df.columns:
        for column in numeric_df.columns:
            report.loc[row, column] = (
                f"Pearson={pearson.loc[row, column]:.4f}, "
                f"Spearman={spearman.loc[row, column]:.4f}"
            )

    return report


def find_correlation_pairs(
    df: pd.DataFrame
) -> list:
    """
    Find correlation pairs excluding self-correlations.
    """

    numeric_df = df.select_dtypes(include=np.number)

    pearson = numeric_df.corr(method="pearson")

    pairs = []

    columns = list(pearson.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            value = pearson.loc[col1, col2]

            pairs.append(
                (col1, col2, value)
            )

    return pairs


def find_dataset() -> str:
    """Find the Epic 2 cleaned dataset."""

    path = (
        "day9_Exploratory Data Analysis (EDA) Deliverable/"
        "data/cleaned_dataset.csv"
    )

    try:
        with open(path, "r", encoding="utf-8"):
            return path
    except FileNotFoundError:
        raise FileNotFoundError(
            "cleaned_dataset.csv was not found."
        )


def main():
    """Generate the correlation report."""

    dataset_path = find_dataset()
    df = pd.read_csv(dataset_path)

    report = correlation_matrix_report(df)

    pairs = find_correlation_pairs(df)

    # Sort by absolute Pearson correlation
    pairs_sorted = sorted(
        pairs,
        key=lambda item: abs(item[2]),
        reverse=True
    )

    strongest = pairs_sorted[:2]
    weakest = pairs_sorted[-2:]

    print("TASK 7: CORRELATION MATRIX REPORT")
    print("=" * 70)

    print("\nNumeric columns:")
    print(list(df.select_dtypes(include=np.number).columns))

    print("\nCorrelation Matrix:")
    print("-" * 70)
    print(report.to_string())

    print("\n2 STRONGEST CORRELATED PAIRS")
    print("-" * 70)

    for col1, col2, value in strongest:
        print(
            f"{col1} <-> {col2}: "
            f"Pearson correlation = {value:.4f}"
        )

    print("\n2 WEAKEST CORRELATED PAIRS")
    print("-" * 70)

    for col1, col2, value in weakest:
        print(
            f"{col1} <-> {col2}: "
            f"Pearson correlation = {value:.4f}"
        )


if __name__ == "__main__":
    main()