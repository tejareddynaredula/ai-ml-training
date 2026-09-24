"""
Day 14 - Advanced Visualization with Seaborn

This module provides advanced statistical visualizations using Seaborn:
1. Correlation heatmap
2. Pairplot
3. Violin plot
4. Regression scatter plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str
) -> None:
    """
    Create an annotated correlation heatmap
    for numeric columns.
    """

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


def plot_pairplot(
    df: pd.DataFrame,
    hue_col: str,
    save_path: str
) -> None:
    """
    Create a pairplot for numeric columns
    with categorical hue coloring.
    """

    if hue_col not in df.columns:
        raise ValueError(
            f"Column '{hue_col}' not found."
        )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_columns:
        raise ValueError("No numeric columns found.")

    plot_data = df[
        numeric_columns + [hue_col]
    ].dropna()

    if plot_data.empty:
        raise ValueError("No valid data available.")

    sns.set_theme(style="whitegrid")

    grid = sns.pairplot(
        plot_data,
        vars=numeric_columns,
        hue=hue_col
    )

    grid.fig.suptitle(
        "Numeric Variables Pairplot",
        y=1.02
    )

    grid.fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(grid.fig)


def plot_violin_by_category(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str,
    save_path: str
) -> None:
    """
    Create a violin plot showing the distribution
    of a numeric column across categories.
    """

    if num_col not in df.columns:
        raise ValueError(
            f"Column '{num_col}' not found."
        )

    if cat_col not in df.columns:
        raise ValueError(
            f"Column '{cat_col}' not found."
        )

    plot_data = df[
        [num_col, cat_col]
    ].dropna()

    if plot_data.empty:
        raise ValueError("No valid data available.")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))

    sns.violinplot(
        data=plot_data,
        x=cat_col,
        y=num_col
    )

    plt.title(
        f"{num_col} Distribution by {cat_col}"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def plot_regression_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: str
) -> None:
    """
    Create a scatter plot with a fitted
    regression line.
    """

    if x_col not in df.columns:
        raise ValueError(
            f"Column '{x_col}' not found."
        )

    if y_col not in df.columns:
        raise ValueError(
            f"Column '{y_col}' not found."
        )

    plot_data = df[
        [x_col, y_col]
    ].dropna()

    if plot_data.empty:
        raise ValueError("No valid data available.")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))

    sns.regplot(
        data=plot_data,
        x=x_col,
        y=y_col
    )

    plt.title(
        f"{y_col} vs {x_col} Regression"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def main():
    """Generate the complete Day 14 Seaborn chart suite."""

    df = pd.read_csv(DATASET_PATH)

    chart_dir = "day14_Seaborn-Advanced\\charts"

    plot_correlation_heatmap(
        df,
        f"{chart_dir}\\final_correlation_heatmap.png"
    )

    plot_pairplot(
        df,
        "Category",
        f"{chart_dir}\\final_numeric_pairplot.png"
    )

    plot_violin_by_category(
        df,
        "Unit_Price",
        "Category",
        f"{chart_dir}\\final_unit_price_violin.png"
    )

    plot_regression_scatter(
        df,
        "Quantity",
        "Unit_Price",
        f"{chart_dir}\\final_quantity_unit_price_regression.png"
    )

    print("Day 14 final deliverable completed successfully.")


if __name__ == "__main__":
    main()