"""
Day 13 - Matplotlib Charts

Task 3: Boxplot
"""

import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_boxplot_by_category(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str,
    save_path: str
) -> None:
    """
    Create and save a boxplot of a numeric column by category.
    """

    if df.empty:
        raise ValueError("Cannot create a boxplot from an empty DataFrame.")

    if num_col not in df.columns:
        raise ValueError(f"Numeric column '{num_col}' not found.")

    if cat_col not in df.columns:
        raise ValueError(f"Category column '{cat_col}' not found.")

    plot_data = df[[num_col, cat_col]].dropna()

    if plot_data.empty:
        raise ValueError("No valid data available for the boxplot.")

    categories = plot_data[cat_col].unique()

    data_by_category = [
        plot_data.loc[
            plot_data[cat_col] == category,
            num_col
        ]
        for category in categories
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        data_by_category,
        tick_labels=categories
    )

    plt.title(f"{num_col} Distribution by {cat_col}")
    plt.xlabel(cat_col)
    plt.ylabel(num_col)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def main():
    """Generate a boxplot using the Epic 2 dataset."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day13_Matplotlib-Charts"
        "\\charts\\unit_price_by_category_boxplot.png"
    )

    plot_boxplot_by_category(
        df,
        "Unit_Price",
        "Category",
        save_path
    )

    print(f"Boxplot saved successfully: {save_path}")


if __name__ == "__main__":
    main()