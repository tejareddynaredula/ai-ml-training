
from pathlib import Path
import json

import pandas as pd


DATA_PATH = (
    Path(__file__).parent.parent
    / "epic2_pandas"
    / "data"
    / "cleaned_dataset.csv"
)

REPORT_PATH = (
    Path(__file__).parent
    / "reports"
    / "final_eda_report.json"
)


def load_dataset():
    return pd.read_csv(DATA_PATH)


def build_pivot_summary(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str
) -> pd.DataFrame:
    return pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=0
    )


def generate_final_report(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")

    category_pivot = build_pivot_summary(
        df, "Category", "Product", "Quantity", "sum"
    )

    product_pivot = build_pivot_summary(
        df, "Product", "Category", "Unit_Price", "mean"
    )

    insights = [
        f"The dataset contains {len(df)} records and "
        f"{len(df.columns)} columns.",

        f"{df['Product'].value_counts().idxmax()} is the most "
        f"frequently occurring product, with "
        f"{df['Product'].value_counts().max()} records.",

        f"{df['Category'].value_counts().idxmax()} is the most "
        f"common category, with "
        f"{df['Category'].value_counts().max()} records.",

        f"The average order quantity is "
        f"{df['Quantity'].mean():.2f} units.",

        f"The correlation between Quantity and Unit_Price "
        f"is {df['Quantity'].corr(df['Unit_Price']):.4f}, "
        f"indicating a weak relationship."
    ]

    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },

        "numeric_statistics": (
            numeric_df.describe()
            .round(2)
            .to_dict()
        ),

        "categorical_counts": {},

        "correlation_matrix": (
            numeric_df.corr()
            .round(4)
            .to_dict()
        ),

        "pivot_tables": {
            "category_wise_quantity": (
                category_pivot.to_dict()
            ),
            "product_wise_average_price": (
                product_pivot.round(2).to_dict()
            )
        },

        "insights": insights
    }

    categorical_columns = df.select_dtypes(
        include=["str", "category"]
    ).columns

    for column in categorical_columns:
        report["categorical_counts"][column] = (
            df[column]
            .value_counts()
            .head(10)
            .to_dict()
        )

    return report


if __name__ == "__main__":
    df = load_dataset()

    report = generate_final_report(df)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    print("Final EDA report generated successfully.")
    print(f"Report saved at: {REPORT_PATH}")