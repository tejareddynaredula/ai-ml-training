
import pandas as pd


def remove_duplicates(
    df: pd.DataFrame,
    subset: list = None
) -> pd.DataFrame:
    """Detect and remove duplicate rows."""

    duplicate_count = df.duplicated(
        subset=subset
    ).sum()

    print("Duplicate Rows Before Removal:", duplicate_count)

    cleaned_df = df.drop_duplicates(
        subset=subset,
        keep="first"
    ).reset_index(drop=True)

    print(
        "Duplicate Rows After Removal:",
        cleaned_df.duplicated(
            subset=subset
        ).sum()
    )

    print("Shape After Removal:", cleaned_df.shape)

    return cleaned_df


def main():
    filepath = (
        "day6_series_dataframes_data_ingestion/data/"
        "ecommerce_orders.csv"
    )

    df = pd.read_csv(filepath)

    # Create duplicate rows for testing
    dirty_df = pd.concat(
        [df, df.iloc[:10]],
        ignore_index=True
    )

    print("Original Shape:", df.shape)
    print("Dirty Shape:", dirty_df.shape)

    print("\nFull-Row Duplicate Removal")
    print("==========================")

    cleaned_df = remove_duplicates(dirty_df)

    print("\nSubset-Based Duplicate Removal")
    print("==============================")

    subset_cleaned_df = remove_duplicates(
        dirty_df,
        subset=["Order_ID"]
    )


if __name__ == "__main__":
    main()