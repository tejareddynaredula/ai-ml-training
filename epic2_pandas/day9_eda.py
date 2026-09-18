
from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).parent.parent
    / "epic2_pandas"
    / "data"
    / "cleaned_dataset.csv"
)


def load_dataset():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def univariate_report(df: pd.DataFrame) -> dict:
    """Generate descriptive statistics for numeric columns."""
    numeric_columns = df.select_dtypes(include="number").columns

    report = {}

    for column in numeric_columns:
        report[column] = {
            "count": int(df[column].count()),
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "minimum": float(df[column].min()),
            "maximum": float(df[column].max()),
            "standard_deviation": float(df[column].std()),
        }

    return report


if __name__ == "__main__":
    df = load_dataset()

    report = univariate_report(df)

    print("Univariate Statistics:")

    for column, statistics in report.items():
        print(f"\n{column}:")
        print(statistics)