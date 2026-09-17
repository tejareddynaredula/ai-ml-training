
import pandas as pd
from pathlib import Path


# ============================================================
# Task 1: Load Dataset
# ============================================================

# Day 7 cleaned dataset location
DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "cleaned_dataset.csv"
)


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the cleaned dataset from CSV."""
    return pd.read_csv(filepath)


# ============================================================
# Task 2: Filtering Operations
# ============================================================

def filter_records(
    df: pd.DataFrame,
    conditions: dict
) -> pd.DataFrame:
    """
    Filter records using multiple conditions.

    Demonstrates:
    - & : AND conditions
    - | : OR conditions
    - isin() : match multiple values
    - between() : filter a range
    - query() : query-based filtering
    """

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


# ============================================================
# Task 3: Groupby Aggregation
# ============================================================

def group_aggregate(
    df: pd.DataFrame,
    group_cols: list,
    agg_map: dict
) -> pd.DataFrame:
    """Perform grouped aggregation with multiple functions."""

    return (
        df.groupby(group_cols)
        .agg(agg_map)
        .reset_index()
    )


def group_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Perform group-wise normalization using transform."""

    result = df.copy()

    group_mean = (
        result.groupby("Category")["Quantity"]
        .transform("mean")
    )

    group_std = (
        result.groupby("Category")["Quantity"]
        .transform("std")
    )

    result["Quantity_Normalized"] = (
        (result["Quantity"] - group_mean) / group_std
    ).fillna(0)

    return result


# ============================================================
# Task 4: Merge / Join Operations
# ============================================================

def merge_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    on: str,
    how: str
) -> pd.DataFrame:
    """
    Merge datasets using a common column.

    inner: matching rows only
    left: all rows from left dataset
    right: all rows from right dataset
    outer: all rows from both datasets
    """

    return pd.merge(
        df1,
        df2,
        on=on,
        how=how,
        validate="many_to_one"
    )


# ============================================================
# Task 5: Top N Records Per Group
# ============================================================

def top_n_per_group(
    df: pd.DataFrame,
    group_col: str,
    sort_col: str,
    n: int
) -> pd.DataFrame:
    """Return the top N records from each group."""

    return (
        df.sort_values(sort_col, ascending=False)
        .groupby(group_col)
        .head(n)
        .reset_index(drop=True)
    )


# ============================================================
# Task 6: Concatenation
# ============================================================

def concatenate_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame
) -> tuple:
    """
    Demonstrate vertical and horizontal concatenation.

    merge: combines related datasets using a common key.
    join: combines datasets using their indexes.
    concat: stacks datasets vertically or horizontally.
    """

    vertical_concat = pd.concat(
        [df1, df2],
        axis=0,
        ignore_index=True
    )

    horizontal_concat = pd.concat(
        [
            df1.reset_index(drop=True),
            df2.reset_index(drop=True)
        ],
        axis=1
    )

    return vertical_concat, horizontal_concat


# ============================================================
# Task 7: Main Demonstration
# ============================================================

def main():

    df = load_dataset(str(DATA_PATH))

    print("\nOriginal Dataset")
    print("================")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ---------------- Filtering ----------------

    print("\nFiltering Example")
    print("=================")

    conditions = {
        "category": ["Electronics", "Furniture"],
        "min_quantity": 2
    }

    filtered_df = filter_records(df, conditions)

    print(filtered_df.head())
    print(f"Filtered Rows: {len(filtered_df)}")

    # ---------------- Groupby ----------------

    print("\nGrouped Summary Report")
    print("======================")

    summary = group_aggregate(
        df,
        ["Category"],
        {
            "Quantity": ["sum", "mean"],
            "Unit_Price": ["mean", "max"]
        }
    )

    print(summary.to_string(index=False))

    # ---------------- Transform ----------------

    print("\nGroup-wise Normalization")
    print("========================")

    transformed_df = group_transform(df)

    print(
        transformed_df[
            ["Category", "Quantity", "Quantity_Normalized"]
        ].head()
    )

    # ---------------- Lookup Dataset ----------------

    lookup_df = pd.DataFrame({
        "Category": [
            "Electronics",
            "Furniture",
            "Clothing",
            "Books",
            "Grocery"
        ],
        "Department": [
            "Technology",
            "Home",
            "Fashion",
            "Education",
            "Food"
        ]
    })

    print("\nLookup Dataset")
    print("==============")
    print(lookup_df)

    # ---------------- Merge Validation ----------------

    print("\nMerge Validation")
    print("================")

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

    merged_df = merge_datasets(
        df,
        lookup_df,
        on="Category",
        how="left"
    )

    print("\nMerged Dataset Preview")
    print(merged_df.head())

    # ---------------- Top N ----------------

    print("\nTop 3 Orders Per Customer")
    print("=========================")

    top_orders = top_n_per_group(
        df,
        "Customer",
        "Quantity",
        3
    )

    print(
        top_orders.head(15).to_string(index=False)
    )

    # ---------------- Concat ----------------

    print("\nConcatenation Example")
    print("=====================")

    sample_1 = df.head(3)
    sample_2 = df.iloc[3:6]

    vertical_df, horizontal_df = concatenate_datasets(
        sample_1,
        sample_2
    )

    print(
        f"Vertical Concat Shape: {vertical_df.shape}"
    )

    print(
        f"Horizontal Concat Shape: {horizontal_df.shape}"
    )


if __name__ == "__main__":
    main()