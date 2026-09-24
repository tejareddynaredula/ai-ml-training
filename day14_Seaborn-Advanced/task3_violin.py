"""
Day 14 - Seaborn Advanced Visualization

Task 3: Violin Plot
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_violin_by_category(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str,
    save_path: str
) -> None:
    """
    Create and save a violin plot showing the distribution
    of a numeric column by category.
    """

    if df.empty:
        raise ValueError("Cannot create a violin plot from an empty DataFrame.")

    if num_col not in df.columns:
        raise ValueError(f"Numeric column '{num_col}' not found.")

    if cat_col not in df.columns:
        raise ValueError(f"Category column '{cat_col}' not found.")

    plot_data = df[[num_col, cat_col]].dropna()

    if plot_data.empty:
        raise ValueError("No valid data available for the violin plot.")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))

    sns.violinplot(
        data=plot_data,
        x=cat_col,
        y=num_col
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
    """Generate a violin plot using the Epic 2 dataset."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day14_Seaborn-Advanced"
        "\\charts\\unit_price_violin.png"
    )

    plot_violin_by_category(
        df,
        "Unit_Price",
        "Category",
        save_path
    )

    print(f"Violin plot saved successfully: {save_path}")


if __name__ == "__main__":
    main()