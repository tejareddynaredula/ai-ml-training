
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


def concatenate_datasets(df1, df2):
    """Perform vertical and horizontal concatenation."""
    vertical_concat = pd.concat(
        [df1, df2],
        axis=0,
        ignore_index=True
    )

    horizontal_concat = pd.concat(
        [df1, df2],
        axis=1
    )

    return vertical_concat, horizontal_concat


if __name__ == "__main__":
    df = load_dataset()

    # Select sample records
    df1 = df.head(3)
    df2 = df.tail(3)

    vertical, horizontal = concatenate_datasets(df1, df2)

    print("Dataset 1 Shape:", df1.shape)
    print("Dataset 2 Shape:", df2.shape)

    print("\nVertical Concatenation:")
    print(vertical)
    print("Vertical Shape:", vertical.shape)

    print("\nHorizontal Concatenation:")
    print(horizontal)
    print("Horizontal Shape:", horizontal.shape)