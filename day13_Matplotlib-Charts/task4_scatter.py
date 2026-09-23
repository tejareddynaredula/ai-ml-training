"""
Day 13 - Matplotlib Charts

Task 4: Scatter Plot
"""

import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_scatter_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: str
) -> None:
    """
    Create and save a scatter plot between two numeric columns.
    """

    if df.empty:
        raise ValueError("Cannot create a scatter plot from an empty DataFrame.")

    if x_col not in df.columns:
        raise ValueError(f"X column '{x_col}' not found.")

    if y_col not in df.columns:
        raise ValueError(f"Y column '{y_col}' not found.")

    plot_data = df[[x_col, y_col]].dropna()

    if plot_data.empty:
        raise ValueError("No valid data available for the scatter plot.")

    plt.figure(figsize=(8, 5))

    plt.scatter(
        plot_data[x_col],
        plot_data[y_col]
    )

    plt.title(f"{y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def main():
    """Generate a scatter plot using the Epic 2 dataset."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day13_Matplotlib-Charts"
        "\\charts\\quantity_vs_unit_price_scatter.png"
    )

    plot_scatter_correlation(
        df,
        "Quantity",
        "Unit_Price",
        save_path
    )

    print(f"Scatter plot saved successfully: {save_path}")


if __name__ == "__main__":
    main()