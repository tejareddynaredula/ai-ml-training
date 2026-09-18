
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


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation between numeric columns."""
    numeric_data = df.select_dtypes(include="number")

    return numeric_data.corr()


if __name__ == "__main__":
    df = load_dataset()

    correlation = correlation_matrix(df)

    print("Correlation Matrix:")
    print(correlation)