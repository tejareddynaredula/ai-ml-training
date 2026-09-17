
from pathlib import Path

import pandas as pd


# Load cleaned dataset
DATA_PATH = (
    Path(__file__).parent.parent
    / "epic2_pandas"
    / "data"
    / "cleaned_dataset.csv"
)


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the cleaned dataset."""
    return pd.read_csv(filepath)


def top_n_per_group(
    df: pd.DataFrame,
    group_col: str,
    sort_col: str,
    n: int
) -> pd.DataFrame:
    """Return top N records for each group."""

    return (
        df.sort_values(sort_col, ascending=False)
        .groupby(group_col)
        .head(n)
        .reset_index(drop=True)
    )


def main():
    df = load_dataset(DATA_PATH)

    top_orders = top_n_per_group(
        df,
        group_col="Customer",
        sort_col="Quantity",
        n=3
    )

    print("Task 5: Top N Records Per Group")
    print("===============================")
    print("\nTop 3 Orders Per Customer:")
    print(top_orders)


if __name__ == "__main__":
    main()