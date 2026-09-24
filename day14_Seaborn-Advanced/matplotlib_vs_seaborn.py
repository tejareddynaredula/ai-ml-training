"""
Day 14 - Matplotlib vs Seaborn Comparison

Compares a category count bar chart created using
Matplotlib and Seaborn.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def create_comparison(df: pd.DataFrame, save_path: str) -> None:
    """
    Create a side-by-side comparison of Matplotlib and Seaborn
    category count charts.
    """

    if df.empty:
        raise ValueError("Cannot create comparison from empty data.")

    if "Category" not in df.columns:
        raise ValueError("Category column not found.")

    plot_data = df[["Category"]].dropna()

    if plot_data.empty:
        raise ValueError("No valid category data available.")

    counts = plot_data["Category"].value_counts()

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # Matplotlib chart
    axes[0].bar(
        counts.index.astype(str),
        counts.values,
        edgecolor="black"
    )

    axes[0].set_title("Matplotlib Bar Chart")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Count")

    # Seaborn chart
    sns.countplot(
        data=plot_data,
        x="Category",
        ax=axes[1]
    )

    axes[1].set_title("Seaborn Countplot")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Count")

    fig.suptitle(
        "Matplotlib vs Seaborn",
        fontsize=16
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    """Generate the Matplotlib vs Seaborn comparison."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day14_Seaborn-Advanced"
        "\\charts\\matplotlib_vs_seaborn.png"
    )

    create_comparison(
        df,
        save_path
    )

    print(
        f"Comparison chart saved successfully: {save_path}"
    )


if __name__ == "__main__":
    main()