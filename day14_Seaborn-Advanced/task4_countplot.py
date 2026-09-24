"""
Day 14 - Seaborn Advanced Visualization

Task 4: Countplot
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_category_count(
    df: pd.DataFrame,
    cat_col: str,
    save_path: str
) -> None:
    """
    Create and save a Seaborn countplot for a categorical column.
    """

    if df.empty:
        raise ValueError("Cannot create a countplot from an empty DataFrame.")

    if cat_col not in df.columns:
        raise ValueError(f"Category column '{cat_col}' not found.")

    plot_data = df[[cat_col]].dropna()

    if plot_data.empty:
        raise ValueError("No valid category data available.")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=plot_data,
        x=cat_col
    )

    plt.title(f"Count of {cat_col}")
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
    """Generate a countplot using the Epic 2 dataset."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day14_Seaborn-Advanced"
        "\\charts\\category_countplot.png"
    )

    plot_category_count(
        df,
        "Category",
        save_path
    )

    print(f"Countplot saved successfully: {save_path}")


if __name__ == "__main__":
    main()