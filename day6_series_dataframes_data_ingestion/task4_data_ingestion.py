import numpy as np
import pandas as pd


DATA_FOLDER = "day6_series_dataframes_data_ingestion/data"
N_ROWS = 150
RANDOM_SEED = 42


def create_dataset():
    np.random.seed(RANDOM_SEED)

    products = ["Laptop", "Phone", "Tablet", "Headphones", "Keyboard"]
    categories = ["Electronics", "Accessories"]

    data = {
        "Order_ID": range(1001, 1001 + N_ROWS),
        "Customer": [f"Customer_{i}" for i in range(1, N_ROWS + 1)],
        "Product": np.random.choice(products, N_ROWS),
        "Category": np.random.choice(categories, N_ROWS),
        "Quantity": np.random.randint(1, 6, N_ROWS),
        "Unit_Price": np.random.randint(500, 50001, N_ROWS),
        "Order_Date": pd.date_range(
            start="2026-01-01",
            periods=N_ROWS,
            freq="D",
        ),
    }

    return pd.DataFrame(data)


def save_datasets(df):
    df.to_csv(
        f"{DATA_FOLDER}/ecommerce_orders.csv",
        index=False,
    )

    df.to_excel(
        f"{DATA_FOLDER}/ecommerce_orders.xlsx",
        index=False,
    )

    df.to_json(
        f"{DATA_FOLDER}/ecommerce_orders.json",
        orient="records",
        date_format="iso",
    )


def main():
    df = create_dataset()
    save_datasets(df)

    print("Canonical E-commerce Dataset")
    print("============================")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nFiles created:")
    print("ecommerce_orders.csv")
    print("ecommerce_orders.xlsx")
    print("ecommerce_orders.json")


if __name__ == "__main__":
    main()