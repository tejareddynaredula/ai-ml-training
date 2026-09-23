"""
Day 13 - Matplotlib Charts

Task 2: Histogram
"""

import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


def plot_histogram(
    data: pd.Series,
    column_name: str,
    save_path: str
) -> None:
    """
    Create and save a histogram for a numeric Series.
    """

    if data.dropna().empty:
        raise ValueError("Cannot create a histogram from an empty series.")

    plt.figure(figsize=(8, 5))

    plt.hist(
        data.dropna(),
        bins=10,
        edgecolor="black"
    )

    plt.title(f"Distribution of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def main():
    """Generate a histogram using the Epic 2 dataset."""

    df = pd.read_csv(DATASET_PATH)

    save_path = (
        "day13_Matplotlib-Charts"
        "\\charts\\quantity_histogram.png"
    )

    plot_histogram(
        df["Quantity"],
        "Quantity",
        save_path
    )

    print(f"Histogram saved successfully: {save_path}")


if __name__ == "__main__":
    main()