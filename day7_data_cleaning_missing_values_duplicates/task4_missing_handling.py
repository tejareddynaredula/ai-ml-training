
import pandas as pd
import numpy as np


def inject_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing values for testing."""

    dirty_df = df.copy()

    dirty_df.loc[0, "Customer"] = np.nan
    dirty_df.loc[1, "Product"] = np.nan
    dirty_df.loc[2, "Quantity"] = np.nan
    dirty_df.loc[3, "Unit_Price"] = np.nan
    dirty_df.loc[4, "Category"] = np.nan

    return dirty_df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: dict
) -> pd.DataFrame:
    """Handle missing values using column-specific strategies."""

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
            mode_value = cleaned_df[column].mode()

            if not mode_value.empty:
                cleaned_df[column] = cleaned_df[column].fillna(
                    mode_value.iloc[0]
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


def main():
    filepath = (
        "day6_series_dataframes_data_ingestion/data/"
        "ecommerce_orders.csv"
    )

    df = pd.read_csv(filepath)

    dirty_df = inject_missing_values(df)

    print("Missing Values Before Cleaning")
    print(dirty_df.isnull().sum())

    strategy = {
        "Customer": "mode",
        "Product": "mode",
        "Category": "mode",
        "Quantity": "median",
        "Unit_Price": "mean",
        "Order_Date": "ffill"
    }

    cleaned_df = handle_missing_values(
        dirty_df,
        strategy
    )

    print("\nMissing Values After Cleaning")
    print(cleaned_df.isnull().sum())


if __name__ == "__main__":
    main()