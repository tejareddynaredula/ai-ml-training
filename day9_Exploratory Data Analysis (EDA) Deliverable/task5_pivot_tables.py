
from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).parent.parent
    / "epic2_pandas"
    / "data"
    / "cleaned_dataset.csv"
)


def load_dataset():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def build_pivot_summary(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str
) -> pd.DataFrame:
    """Create a pivot table summary."""
    pivot_table = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=0
    )

    return pivot_table


if __name__ == "__main__":
    df = load_dataset()

    # Pivot Table 1: Category-wise total quantity
    category_summary = build_pivot_summary(
        df,
        index="Category",
        columns="Product",
        values="Quantity",
        aggfunc="sum"
    )

    print("\nCategory-wise Total Quantity:")
    print(category_summary)

    # Pivot Table 2: Product-wise average unit price
    product_summary = build_pivot_summary(
        df,
        index="Product",
        columns="Category",
        values="Unit_Price",
        aggfunc="mean"
    )

    print("\nProduct-wise Average Unit Price:")
    print(product_summary)