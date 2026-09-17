
from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).parent.parent
    / "epic2_pandas"
    / "data"
    / "cleaned_dataset.csv"
)

LOOKUP_PATH = Path(__file__).parent / "category_lookup.csv"


def load_datasets():
    """Load the sales and category lookup datasets."""
    sales_df = pd.read_csv(DATA_PATH)
    lookup_df = pd.read_csv(LOOKUP_PATH)

    return sales_df, lookup_df


def validate_merge(sales_df, lookup_df):
    """Validate merge row counts and prevent unexpected multiplication."""
    merged_df = pd.merge(
        sales_df,
        lookup_df,
        on="Category",
        how="left",
        validate="many_to_one"
    )

    original_rows = len(sales_df)
    merged_rows = len(merged_df)

    print("Original Rows:", original_rows)
    print("Merged Rows:", merged_rows)

    if merged_rows != original_rows:
        print("Warning: Unexpected row multiplication detected.")
    else:
        print("Validation Passed: No unexpected row multiplication.")

    return merged_df


def grouped_summary(merged_df):
    """Create a grouped summary by category and department."""
    summary = (
        merged_df
        .groupby(["Category", "Department"])
        .agg(
            Total_Quantity=("Quantity", "sum"),
            Average_Price=("Unit_Price", "mean"),
            Record_Count=("Order_ID", "count")
        )
        .reset_index()
    )

    return summary


if __name__ == "__main__":
    sales_df, lookup_df = load_datasets()

    merged_df = validate_merge(sales_df, lookup_df)

    summary = grouped_summary(merged_df)

    print("\nGrouped Summary:")
    print(summary)