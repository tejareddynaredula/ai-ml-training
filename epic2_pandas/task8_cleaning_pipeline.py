
import pandas as pd
import numpy as np


def inject_dirty_data(
    df: pd.DataFrame,
    missing_pct: float = 5,
    dup_count: int = 10
) -> pd.DataFrame:
    """Inject missing values and duplicate rows."""

    dirty_df = pd.concat(
        [df, df.iloc[:dup_count]],
        ignore_index=True
    )

    rng = np.random.default_rng(42)
    total_cells = dirty_df.size
    missing_count = int(total_cells * missing_pct / 100)

    for _ in range(missing_count):
        row = rng.integers(0, len(dirty_df))
        col = rng.integers(0, len(dirty_df.columns))
        dirty_df.iloc[row, col] = np.nan

    return dirty_df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: dict
) -> pd.DataFrame:
    """Handle missing values by column strategy."""

    cleaned_df = df.copy()

    for column, method in strategy.items():

        if column not in cleaned_df.columns:
            continue

        if method == "mean":
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].mean()
            )

        elif method == "median":
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].median()
            )

        elif method == "mode":
            mode_values = cleaned_df[column].mode()

            if not mode_values.empty:
                cleaned_df[column] = cleaned_df[column].fillna(
                    mode_values.iloc[0]
                )

        elif method == "ffill":
            cleaned_df[column] = cleaned_df[column].ffill()

        elif method == "bfill":
            cleaned_df[column] = cleaned_df[column].bfill()

        elif method == "drop":
            cleaned_df = cleaned_df.dropna(
                subset=[column]
            )

        else:
            raise ValueError(
                f"Unsupported strategy: {method}"
            )

    return cleaned_df


def remove_duplicates(
    df: pd.DataFrame,
    subset: list = None
) -> pd.DataFrame:
    """Remove duplicate rows."""

    return df.drop_duplicates(
        subset=subset,
        keep="first"
    ).reset_index(drop=True)


def fix_dtypes(
    df: pd.DataFrame,
    dtype_map: dict
) -> pd.DataFrame:
    """Correct column data types."""

    corrected_df = df.copy()

    for column, dtype in dtype_map.items():

        if column not in corrected_df.columns:
            continue

        if dtype == "datetime":
            corrected_df[column] = pd.to_datetime(
                corrected_df[column],
                errors="coerce"
            )

        elif dtype == "numeric":
            corrected_df[column] = pd.to_numeric(
                corrected_df[column],
                errors="coerce"
            )

    return corrected_df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a data quality report."""

    return {
        "shape": df.shape,
        "null_counts": int(df.isnull().sum().sum()),
        "duplicate_counts": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict()
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete data cleaning pipeline."""

    cleaned_df = df.copy()

    # Step 1: Fix data types
    dtype_map = {
        "Order_Date": "datetime",
        "Quantity": "numeric",
        "Unit_Price": "numeric"
    }

    cleaned_df = fix_dtypes(
        cleaned_df,
        dtype_map
    )

    # Step 2: Handle missing values
    strategy = {
        "Customer": "mode",
        "Product": "mode",
        "Category": "mode",
        "Quantity": "median",
        "Unit_Price": "mean",
        "Order_Date": "ffill"
    }

    cleaned_df = handle_missing_values(
        cleaned_df,
        strategy
    )

    # Step 3: Fill any remaining missing values
    cleaned_df = cleaned_df.ffill().bfill()

    # Step 4: Remove duplicates at the end
    cleaned_df = remove_duplicates(cleaned_df)

    return cleaned_df


def main():
    filepath = (
        "day6_series_dataframes_data_ingestion/data/"
        "ecommerce_orders.csv"
    )

    output_path = (
        "epic2_pandas/data/cleaned_dataset.csv"
    )

    df = pd.read_csv(filepath)

    dirty_df = inject_dirty_data(df)

    before_report = data_quality_report(dirty_df)

    cleaned_df = clean_dataset(dirty_df)

    after_report = data_quality_report(cleaned_df)

    cleaned_df.to_csv(
        output_path,
        index=False
    )

    print("Before Cleaning")
    print("===============")
    print(before_report)

    print("\nAfter Cleaning")
    print("==============")
    print(after_report)

    print("\nCleaned Dataset Saved")
    print("====================")
    print(output_path)


if __name__ == "__main__":
    main()