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


def filter_records(
    df: pd.DataFrame,
    conditions: dict
) -> pd.DataFrame:
    """Filter records using multiple conditions."""

    result = df.copy()

    if "category" in conditions:
        result = result[
            result["Category"].isin(conditions["category"])
        ]

    if "min_quantity" in conditions:
        result = result[
            result["Quantity"] >= conditions["min_quantity"]
        ]

    if "max_quantity" in conditions:
        result = result[
            result["Quantity"] <= conditions["max_quantity"]
        ]

    if "min_price" in conditions and "max_price" in conditions:
        result = result[
            result["Unit_Price"].between(
                conditions["min_price"],
                conditions["max_price"]
            )
        ]

    if "customer" in conditions:
        customer = conditions["customer"]
        result = result.query("Customer == @customer")

    return result


def main():
    df = load_dataset(DATA_PATH)

    conditions = {
        "category": ["Electronics", "Accessories"],
        "min_quantity": 2,
        "max_quantity": 5,
        "min_price": 1000,
        "max_price": 50000
    }

    filtered_df = filter_records(df, conditions)

    print("Task 2: Filtering Operations")
    print("============================")
    print(f"Original Rows: {len(df)}")
    print(f"Filtered Rows: {len(filtered_df)}")
    print("\nFiltered Dataset:")
    print(filtered_df.head())


if __name__ == "__main__":
    main()