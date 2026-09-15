import pandas as pd

DATA_FOLDER = "day6_series_dataframes_data_ingestion/data"


def inspect_dataset(name, filepath):
    print(f"\n{'=' * 60}")
    print(f"{name} DATASET")
    print(f"{'=' * 60}")

    df = pd.read_csv(filepath) if filepath.endswith(".csv") else (
        pd.read_excel(filepath)
        if filepath.endswith(".xlsx")
        else pd.read_json(filepath)
    )

    print("\n1. HEAD:")
    print(df.head())

    print("\n2. TAIL:")
    print(df.tail())

    print("\n3. INFO:")
    df.info()

    print("\n4. DESCRIBE:")
    print(df.describe())

    print("\n5. DTYPES:")
    print(df.dtypes)

    print("\n6. COLUMNS:")
    print(df.columns.tolist())

    print("\n7. SHAPE:")
    print(df.shape)


def main():
    inspect_dataset(
        "CSV",
        f"{DATA_FOLDER}/ecommerce_orders.csv"
    )

    inspect_dataset(
        "EXCEL",
        f"{DATA_FOLDER}/ecommerce_orders.xlsx"
    )

    inspect_dataset(
        "JSON",
        f"{DATA_FOLDER}/ecommerce_orders.json"
    )


if __name__ == "__main__":
    main()