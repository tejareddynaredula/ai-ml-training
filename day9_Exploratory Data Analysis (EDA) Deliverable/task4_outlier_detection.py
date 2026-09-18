
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


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """Return rows identified as outliers using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower_limit)
        | (df[column] > upper_limit)
    ]

    return outliers


if __name__ == "__main__":
    df = load_dataset()

    numeric_columns = ["Quantity", "Unit_Price"]

    for column in numeric_columns:
        outliers = detect_outliers_iqr(df, column)

        print(f"\nColumn: {column}")
        print("Outlier Count:", len(outliers))
        print(outliers[[column]])