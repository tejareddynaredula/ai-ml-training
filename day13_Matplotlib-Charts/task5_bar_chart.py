"""
Day 13 - Matplotlib Charts

Task 5: Bar Chart
"""

import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_category_counts(
    df: pd.DataFrame,
    cat_col: str,
    save_path: str
) -> None:
    """
    Create and save a bar chart showing category counts.
    """

    if df.empty:
        raise ValueError("Cannot create a bar chart from an empty DataFrame.")

    if cat_col not in df.columns:
        raise ValueError(f"Category column '{cat_col}' not found.")

    counts = df[cat_col].dropna().value_counts()

    if counts.empty:
        raise ValueError("No valid category data available.")

    plt.figure(figsize=(8, 5))

    plt.bar(
        counts.index.astype(str),
        counts.values,
        edgecolor="black"
    )

    plt.title(f"Count by {cat_col}")
    plt.xlabel(cat_col)
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def main():
    """Generate a category count bar chart using the Epic 2 dataset."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day13_Matplotlib-Charts"
        "\\charts\\category_counts_bar_chart.png"
    )

    plot_category_counts(
        df,
        "Category",
        save_path
    )

    print(f"Bar chart saved successfully: {save_path}")


if __name__ == "__main__":
    main()