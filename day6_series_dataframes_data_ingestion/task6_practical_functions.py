import pandas as pd

DATA_FOLDER = "day6_series_dataframes_data_ingestion/data"


def load_dataset(filepath: str) -> pd.DataFrame:
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        return pd.read_excel(filepath)
    elif filepath.endswith(".json"):
        return pd.read_json(filepath)
    else:
        raise ValueError("Unsupported file format")


def dataframe_summary(df) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "memory_usage": int(df.memory_usage(deep=True).sum())
    }


def select_columns_rows(df) -> None:
    print("\nColumn Selection:")
    print(df[["Customer", "Product", "Unit_Price"]].head())

    print("\nUsing loc:")
    print(df.loc[0:4, ["Customer", "Product", "Unit_Price"]])

    print("\nUsing iloc:")
    print(df.iloc[0:5, 1:4])

    print("\nRow Slicing:")
    print(df[10:15])


def main():
    filepath = f"{DATA_FOLDER}/ecommerce_orders.csv"

    df = load_dataset(filepath)

    print("Loaded Dataset")
    print("================")
    print(df.head())

    print("\nDataFrame Summary:")
    summary = dataframe_summary(df)

    for key, value in summary.items():
        print(f"{key}: {value}")

    select_columns_rows(df)


if __name__ == "__main__":
    main()