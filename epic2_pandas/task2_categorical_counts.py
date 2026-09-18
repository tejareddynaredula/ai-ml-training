
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


def categorical_value_counts(df: pd.DataFrame) -> dict:
    """Generate value counts for categorical columns."""
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    report = {}

    for column in categorical_columns:
        report[column] = df[column].value_counts().to_dict()

    return report


if __name__ == "__main__":
    df = load_dataset()

    report = categorical_value_counts(df)

    print("Categorical Value Counts:")

    for column, counts in report.items():
        print(f"\n{column}:")
        print(counts)