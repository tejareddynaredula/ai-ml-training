
import pandas as pd


def fix_dtypes(
    df: pd.DataFrame,
    dtype_map: dict
) -> pd.DataFrame:
    """Convert columns to the required data types."""

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

        else:
            corrected_df[column] = corrected_df[column].astype(
                dtype
            )

    return corrected_df


def main():
    filepath = (
        "day6_series_dataframes_data_ingestion/data/"
        "ecommerce_orders.csv"
    )

    df = pd.read_csv(filepath)

    # Intentionally change numeric columns to strings
    df["Quantity"] = df["Quantity"].astype(str)
    df["Unit_Price"] = df["Unit_Price"].astype(str)

    print("Data Types Before Correction")
    print("============================")
    print(df.dtypes)

    dtype_map = {
        "Order_Date": "datetime",
        "Quantity": "numeric",
        "Unit_Price": "numeric"
    }

    corrected_df = fix_dtypes(
        df,
        dtype_map
    )

    print("\nData Types After Correction")
    print("===========================")
    print(corrected_df.dtypes)

    print("\nData Type Validation")
    print("====================")
    print(
        "Order_Date is datetime:",
        pd.api.types.is_datetime64_any_dtype(
            corrected_df["Order_Date"]
        )
    )

    print(
        "Quantity is numeric:",
        pd.api.types.is_numeric_dtype(
            corrected_df["Quantity"]
        )
    )

    print(
        "Unit_Price is numeric:",
        pd.api.types.is_numeric_dtype(
            corrected_df["Unit_Price"]
        )
    )


if __name__ == "__main__":
    main()