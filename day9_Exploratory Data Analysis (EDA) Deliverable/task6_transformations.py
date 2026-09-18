
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


def categorize_column(
    df: pd.DataFrame,
    column: str,
    bins: list,
    labels: list
) -> pd.DataFrame:
    """Categorize a numeric column using pd.cut."""
    result = df.copy()

    result[f"{column}_Category"] = pd.cut(
        result[column],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return result


def quantity_description(quantity):
    """Describe the quantity using apply()."""
    if quantity >= 4:
        return "High"
    elif quantity >= 2:
        return "Medium"
    else:
        return "Low"


if __name__ == "__main__":
    df = load_dataset()

    # Apply: Create quantity description
    df["Quantity_Level"] = df["Quantity"].apply(
        quantity_description
    )

    # Map: Convert category names to short codes
    category_mapping = {
        "Electronics": "ELEC",
        "Accessories": "ACC"
    }

    df["Category_Code"] = df["Category"].map(
        category_mapping
    )

    # pd.cut: Categorize unit prices
    price_bins = [0, 10000, 25000, 40000, float("inf")]
    price_labels = ["Low", "Medium", "High", "Very High"]

    df = categorize_column(
        df,
        column="Unit_Price",
        bins=price_bins,
        labels=price_labels
    )

    print("\nTransformed Dataset:")
    print(
        df[
            [
                "Quantity",
                "Quantity_Level",
                "Category",
                "Category_Code",
                "Unit_Price",
                "Unit_Price_Category"
            ]
        ].head(10)
    )

    print("\nQuantity Level Counts:")
    print(df["Quantity_Level"].value_counts())

    print("\nCategory Code Counts:")
    print(df["Category_Code"].value_counts())

    print("\nUnit Price Category Counts:")
    print(df["Unit_Price_Category"].value_counts())