
import pandas as pd
import numpy as np


def inject_dirty_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing values and duplicate rows."""

    dirty_df = df.copy()

    # Add duplicate rows
    duplicates = dirty_df.iloc[:10].copy()
    dirty_df = pd.concat(
        [dirty_df, duplicates],
        ignore_index=True
    )

    # Add missing values
    dirty_df.loc[0, "Customer"] = np.nan
    dirty_df.loc[1, "Product"] = np.nan
    dirty_df.loc[2, "Quantity"] = np.nan
    dirty_df.loc[3, "Unit_Price"] = np.nan
    dirty_df.loc[4, "Category"] = np.nan

    return dirty_df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a data quality report."""

    return {
        "shape": df.shape,
        "null_counts": int(df.isnull().sum().sum()),
        "duplicate_counts": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict()
    }


def print_quality_report(
    title: str,
    report: dict
) -> None:
    """Print a formatted quality report."""

    print(f"\n{title}")
    print("=" * len(title))

    print("Shape:", report["shape"])
    print("Total Missing Values:", report["null_counts"])
    print("Duplicate Rows:", report["duplicate_counts"])

    print("Data Types:")
    for column, dtype in report["dtypes"].items():
        print(f"  {column}: {dtype}")


def main():
    filepath = (
        "day6_series_dataframes_data_ingestion/data/"
        "ecommerce_orders.csv"
    )

    df = pd.read_csv(filepath)

    dirty_df = inject_dirty_data(df)

    before_report = data_quality_report(dirty_df)

    # Basic cleaning for report comparison
    cleaned_df = dirty_df.drop_duplicates().copy()

    cleaned_df["Customer"] = cleaned_df["Customer"].fillna(
        cleaned_df["Customer"].mode().iloc[0]
    )

    cleaned_df["Product"] = cleaned_df["Product"].fillna(
        cleaned_df["Product"].mode().iloc[0]
    )

    cleaned_df["Category"] = cleaned_df["Category"].fillna(
        cleaned_df["Category"].mode().iloc[0]
    )

    cleaned_df["Quantity"] = cleaned_df["Quantity"].fillna(
        cleaned_df["Quantity"].median()
    )

    cleaned_df["Unit_Price"] = cleaned_df["Unit_Price"].fillna(
        cleaned_df["Unit_Price"].mean()
    )

    cleaned_df["Order_Date"] = pd.to_datetime(
        cleaned_df["Order_Date"],
        errors="coerce"
    )

    cleaned_df["Order_Date"] = cleaned_df["Order_Date"].ffill()

    after_report = data_quality_report(cleaned_df)

    print_quality_report(
        "Before Cleaning",
        before_report
    )

    print_quality_report(
        "After Cleaning",
        after_report
    )


if __name__ == "__main__":
    main()