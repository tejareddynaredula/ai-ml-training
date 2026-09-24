"""
Day 14 - Seaborn Advanced Visualization

Task 1: Correlation Heatmap
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str
) -> None:
    """
    Create and save an annotated correlation heatmap
    for numeric columns.
    """

    if df.empty:
        raise ValueError("Cannot create a heatmap from an empty DataFrame.")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("No numeric columns found.")

    correlation_matrix = numeric_df.corr()

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def main():
    """Generate a correlation heatmap using the Epic 2 dataset."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day14_Seaborn-Advanced"
        "\\charts\\correlation_heatmap.png"
    )

    plot_correlation_heatmap(
        df,
        save_path
    )

    print(f"Correlation heatmap saved successfully: {save_path}")


if __name__ == "__main__":
    main()