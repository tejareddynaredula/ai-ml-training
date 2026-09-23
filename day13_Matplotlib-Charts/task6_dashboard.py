"""
Day 13 - Matplotlib Charts

Task 6: 4-Chart Dashboard
"""

import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_dashboard_grid(
    df: pd.DataFrame,
    save_path: str
) -> None:
    """
    Create and save a 2x2 dashboard containing:
    1. Histogram
    2. Boxplot
    3. Scatter plot
    4. Bar chart
    """

    if df.empty:
        raise ValueError("Cannot create a dashboard from an empty DataFrame.")

    required_columns = [
        "Quantity",
        "Unit_Price",
        "Category"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found.")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9)
    )

    # 1. Histogram
    axes[0, 0].hist(
        df["Quantity"].dropna(),
        bins=10,
        edgecolor="black"
    )

    axes[0, 0].set_title("Quantity Distribution")
    axes[0, 0].set_xlabel("Quantity")
    axes[0, 0].set_ylabel("Frequency")

    # 2. Boxplot
    box_data = [
        df.loc[
            df["Category"] == category,
            "Unit_Price"
        ].dropna()
        for category in df["Category"].dropna().unique()
    ]

    categories = df["Category"].dropna().unique()

    axes[0, 1].boxplot(
        box_data,
        tick_labels=categories
    )

    axes[0, 1].set_title("Unit Price by Category")
    axes[0, 1].set_xlabel("Category")
    axes[0, 1].set_ylabel("Unit Price")

    # 3. Scatter plot
    scatter_data = df[
        ["Quantity", "Unit_Price"]
    ].dropna()

    axes[1, 0].scatter(
        scatter_data["Quantity"],
        scatter_data["Unit_Price"]
    )

    axes[1, 0].set_title("Quantity vs Unit Price")
    axes[1, 0].set_xlabel("Quantity")
    axes[1, 0].set_ylabel("Unit Price")

    # 4. Bar chart
    category_counts = df["Category"].dropna().value_counts()

    axes[1, 1].bar(
        category_counts.index.astype(str),
        category_counts.values,
        edgecolor="black"
    )

    axes[1, 1].set_title("Count by Category")
    axes[1, 1].set_xlabel("Category")
    axes[1, 1].set_ylabel("Count")

    # Shared dashboard title
    fig.suptitle(
        "Epic 2 Data Visualization Dashboard",
        fontsize=16
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    """Generate the 4-chart dashboard."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day13_Matplotlib-Charts"
        "\\charts\\day13_dashboard.png"
    )

    plot_dashboard_grid(
        df,
        save_path
    )

    print(f"Dashboard saved successfully: {save_path}")


if __name__ == "__main__":
    main()