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


def group_aggregate(
    df: pd.DataFrame,
    group_cols,
    agg_map: dict
) -> pd.DataFrame:
    """Perform grouped aggregation with multiple functions."""

    return (
        df.groupby(group_cols)
        .agg(agg_map)
        .reset_index()
    )


def group_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Perform group-wise quantity normalization."""

    result = df.copy()

    group_mean = result.groupby("Category")["Quantity"].transform("mean")
    group_std = result.groupby("Category")["Quantity"].transform("std")

    result["Quantity_Normalized"] = (
        (result["Quantity"] - group_mean) / group_std
    )

    return result


def main():
    df = load_dataset(DATA_PATH)

    # Groupby aggregation
    agg_map = {
        "Quantity": ["sum", "mean"],
        "Unit_Price": ["mean", "max"]
    }

    summary = group_aggregate(
        df,
        ["Category"],
        agg_map
    )

    print("Task 3: Groupby Aggregation")
    print("===========================")
    print("\nGrouped Summary:")
    print(summary)

    # Group-wise transformation
    normalized_df = group_transform(df)

    print("\nGroup-wise Normalization:")
    print(
        normalized_df[
            ["Category", "Quantity", "Quantity_Normalized"]
        ].head()
    )


if __name__ == "__main__":
    main()