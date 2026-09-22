"""
Day 12 - Correlation & Distribution Concepts

Main deliverable containing:
1. Pearson correlation from scratch
2. Correlation matrix report
3. Normal distribution fitting
4. Z-score calculation and outlier detection
"""

import numpy as np
import pandas as pd
from scipy import stats


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient from scratch.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if len(x) < 2:
        raise ValueError("At least two observations are required.")

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))

    denominator = np.sqrt(
        np.sum((x - x_mean) ** 2)
        * np.sum((y - y_mean) ** 2)
    )

    if denominator == 0:
        raise ValueError("Correlation is undefined for constant data.")

    return float(numerator / denominator)


def correlation_matrix_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a correlation report for numeric columns.

    The report contains both Pearson and Spearman
    correlation values for every pair of numeric columns.
    """
    numeric_df = df.select_dtypes(include=np.number)

    columns = numeric_df.columns
    report = pd.DataFrame(
        index=columns,
        columns=columns,
        dtype=object
    )

    for col1 in columns:
        for col2 in columns:
            pearson = pearson_correlation(
                numeric_df[col1].values,
                numeric_df[col2].values
            )

            spearman = stats.spearmanr(
                numeric_df[col1].values,
                numeric_df[col2].values
            ).statistic

            report.loc[col1, col2] = (
                f"Pearson={pearson:.4f}, "
                f"Spearman={spearman:.4f}"
            )

    return report


def fit_normal_distribution(data: np.ndarray) -> dict:
    """
    Fit a normal distribution to the supplied data.

    Returns:
        dict containing mean, standard deviation,
        Shapiro-Wilk statistic and p-value.
    """
    data = np.asarray(data, dtype=float)

    mean = float(np.mean(data))
    std = float(np.std(data))

    shapiro_result = stats.shapiro(data)

    return {
        "mean": mean,
        "std": std,
        "shapiro_statistic": float(shapiro_result.statistic),
        "shapiro_p_value": float(shapiro_result.pvalue),
    }


def compute_zscores(data: np.ndarray) -> np.ndarray:
    """
    Calculate z-scores from scratch using population standard deviation.
    """
    data = np.asarray(data, dtype=float)

    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        raise ValueError("Z-scores cannot be calculated for constant data.")

    return (data - mean) / std


def identify_outliers(
    data: np.ndarray,
    threshold: float = 2.0
) -> np.ndarray:
    """
    Identify values whose absolute z-score is greater
    than the supplied threshold.
    """
    data = np.asarray(data, dtype=float)

    zscores = compute_zscores(data)

    return data[np.abs(zscores) > threshold]


def main():
    """
    Demonstrate the Day 12 functionality using the Epic 2 dataset.
    """

    dataset_path = (
        "day9_Exploratory Data Analysis (EDA) Deliverable"
        "\\data\\cleaned_dataset.csv"
    )

    df = pd.read_csv(dataset_path)

    print("=" * 70)
    print("DAY 12 - CORRELATION & DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Correlation matrix
    # ---------------------------------------------------------

    print("\n1. CORRELATION MATRIX")
    print("-" * 70)

    correlation_report = correlation_matrix_report(df)

    print(correlation_report)

    # ---------------------------------------------------------
    # 2. Find strongest and weakest correlation pairs
    # ---------------------------------------------------------

    numeric_df = df.select_dtypes(include=np.number)

    pairs = []

    for i, col1 in enumerate(numeric_df.columns):
        for j, col2 in enumerate(numeric_df.columns):
            if i < j:
                correlation = pearson_correlation(
                    numeric_df[col1].values,
                    numeric_df[col2].values
                )

                pairs.append(
                    (col1, col2, correlation)
                )

    strongest = sorted(
        pairs,
        key=lambda item: abs(item[2]),
        reverse=True
    )[:2]

    weakest = sorted(
        pairs,
        key=lambda item: abs(item[2])
    )[:2]

    print("\nStrongest correlation pairs:")
    for col1, col2, correlation in strongest:
        print(
            f"{col1} <-> {col2}: "
            f"Pearson = {correlation:.4f}"
        )

    print("\nWeakest correlation pairs:")
    for col1, col2, correlation in weakest:
        print(
            f"{col1} <-> {col2}: "
            f"Pearson = {correlation:.4f}"
        )

    # ---------------------------------------------------------
    # 3. Normality analysis
    # ---------------------------------------------------------

    print("\n2. NORMALITY ANALYSIS")
    print("-" * 70)

    for column in ["Quantity", "Unit_Price"]:
        result = fit_normal_distribution(
            df[column].dropna().values
        )

        print(f"\nColumn: {column}")
        print(f"Mean: {result['mean']:.4f}")
        print(f"Standard deviation: {result['std']:.4f}")
        print(
            f"Shapiro statistic: "
            f"{result['shapiro_statistic']:.6f}"
        )
        print(
            f"Shapiro p-value: "
            f"{result['shapiro_p_value']:.6f}"
        )

        if result["shapiro_p_value"] > 0.05:
            print(
                "Interpretation: No strong evidence "
                "against normality."
            )
        else:
            print(
                "Interpretation: Evidence suggests "
                "the data is not normally distributed."
            )

    # ---------------------------------------------------------
    # 4. Z-score and outlier analysis
    # ---------------------------------------------------------

    print("\n3. Z-SCORE / OUTLIER ANALYSIS")
    print("-" * 70)

    data = np.array(
        [
            10, 11, 12, 12, 13,
            13, 14, 14, 15, 15,
            16, 17, 18, 19, 50
        ],
        dtype=float
    )

    zscores = compute_zscores(data)

    print(f"Mean: {np.mean(data):.4f}")
    print(f"Standard deviation: {np.std(data):.4f}")

    print("\nValues beyond +/-2 standard deviations:")
    print(identify_outliers(data, threshold=2))

    print("\nValues beyond +/-3 standard deviations:")
    print(identify_outliers(data, threshold=3))


if __name__ == "__main__":
    main()