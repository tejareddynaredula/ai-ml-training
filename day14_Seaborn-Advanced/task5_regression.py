"""
Day 14 - Seaborn Advanced Visualization

Task 5: Regression Plot
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_regression_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: str
) -> None:
    """
    Create and save a scatter plot with a fitted regression line.
    """

    if df.empty:
        raise ValueError(
            "Cannot create a regression plot from an empty DataFrame."
        )

    if x_col not in df.columns:
        raise ValueError(f"X column '{x_col}' not found.")

    if y_col not in df.columns:
        raise ValueError(f"Y column '{y_col}' not found.")

    plot_data = df[[x_col, y_col]].dropna()

    if plot_data.empty:
        raise ValueError(
            "No valid data available for the regression plot."
        )

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=plot_data,
        x=x_col,
        y=y_col
    )

    plt.title(f"{y_col} vs {x_col} with Regression Line")
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
    """Generate a regression plot using the Epic 2 dataset."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day14_Seaborn-Advanced"
        "\\charts\\quantity_vs_unit_price_regression.png"
    )

    plot_regression_scatter(
        df,
        "Quantity",
        "Unit_Price",
        save_path
    )

    print(
        f"Regression plot saved successfully: {save_path}"
    )


if __name__ == "__main__":
    main()