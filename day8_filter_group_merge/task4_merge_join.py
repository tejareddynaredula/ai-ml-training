
from pathlib import Path

import pandas as pd


# Load cleaned dataset
DATA_PATH = (
    Path(__file__).parent.parent
    / "epic2_pandas"
    / "data"
    / "cleaned_dataset.csv"
)

LOOKUP_PATH = Path(__file__).parent / "category_lookup.csv"


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a CSV dataset."""
    return pd.read_csv(filepath)


def merge_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    on: str,
    how: str
) -> pd.DataFrame:
    """Merge two datasets using a common column."""

    return pd.merge(
        df1,
        df2,
        on=on,
        how=how,
        validate="many_to_one"
    )


def main():
    df = load_dataset(DATA_PATH)
    lookup_df = load_dataset(LOOKUP_PATH)

    print("Task 4: Merge / Join Operations")
    print("===============================")

    print(f"Original Rows: {len(df)}")

    for join_type in ["inner", "left", "right", "outer"]:
        merged_df = merge_datasets(
            df,
            lookup_df,
            on="Category",
            how=join_type
        )

        print(
            f"{join_type.capitalize()} Join Rows: "
            f"{len(merged_df)}"
        )

    # Validate left join
    left_df = merge_datasets(
        df,
        lookup_df,
        on="Category",
        how="left"
    )

    print("\nMerged Dataset Preview:")
    print(left_df.head())

    print(f"\nMerged Rows: {len(left_df)}")

    if len(left_df) == len(df):
        print("Validation: No unexpected row multiplication")
    else:
        print("Validation: Row count changed")


if __name__ == "__main__":
    main()