
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
    / "eda_report.json"
)


def load_dataset():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def generate_eda_report(df: pd.DataFrame) -> dict:
    """Generate a complete EDA report."""
    numeric_df = df.select_dtypes(include="number")

    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "numeric_statistics": {},
        "categorical_counts": {},
        "correlation_matrix": {},
        "outliers": {}
    }

    # Numeric statistics
    report["numeric_statistics"] = (
        numeric_df.describe()
        .round(2)
        .to_dict()
    )

    # Categorical value counts
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        report["categorical_counts"][column] = (
            df[column]
            .value_counts()
            .head(10)
            .to_dict()
        )

    # Correlation matrix
    report["correlation_matrix"] = (
        numeric_df.corr()
        .round(4)
        .to_dict()
    )

    # IQR outlier detection
    for column in numeric_df.columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        outlier_count = int(
            (
                (df[column] < lower_limit)
                | (df[column] > upper_limit)
            ).sum()
        )

        report["outliers"][column] = {
            "lower_limit": round(float(lower_limit), 2),
            "upper_limit": round(float(upper_limit), 2),
            "outlier_count": outlier_count
        }

    return report


if __name__ == "__main__":
    df = load_dataset()

    report = generate_eda_report(df)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    print("EDA report generated successfully.")
    print(f"Report saved at: {REPORT_PATH}")