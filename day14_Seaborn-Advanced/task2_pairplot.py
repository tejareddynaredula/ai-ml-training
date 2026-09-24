"""
Day 14 - Seaborn Advanced Visualization

Task 2: Pairplot
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_pairplot(
    df: pd.DataFrame,
    hue_col: str,
    save_path: str
) -> None:
    """
    Create and save a Seaborn pairplot for numeric columns,
    with observations colored by a categorical column.
    """

    if df.empty:
        raise ValueError("Cannot create a pairplot from an empty DataFrame.")

    if hue_col not in df.columns:
        raise ValueError(f"Hue column '{hue_col}' not found.")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_columns) < 2:
        raise ValueError("At least two numeric columns are required.")

    plot_data = df[numeric_columns + [hue_col]].dropna()

    if plot_data.empty:
        raise ValueError("No valid data available for the pairplot.")

    sns.set_theme(style="whitegrid")

    pair_plot = sns.pairplot(
        plot_data,
        vars=numeric_columns,
        hue=hue_col
    )

    pair_plot.fig.suptitle(
        "Pairplot of Numeric Variables by Category",
        y=1.02
    )

    pair_plot.fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(pair_plot.fig)


def main():
    """Generate a pairplot using the Epic 2 dataset."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day14_Seaborn-Advanced"
        "\\charts\\numeric_pairplot.png"
    )

    plot_pairplot(
        df,
        "Category",
        save_path
    )

    print(f"Pairplot saved successfully: {save_path}")


if __name__ == "__main__":
    main()