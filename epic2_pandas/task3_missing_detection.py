
import pandas as pd
import numpy as np


def inject_dirty_data(
    df: pd.DataFrame,
    missing_pct: float,
    dup_count: int
) -> pd.DataFrame:
    """Inject missing values and duplicate rows."""

    dirty_df = df.copy()

    # Add duplicate rows
    if dup_count > 0 and len(dirty_df) > 0:
        duplicates = dirty_df.sample(
            n=min(dup_count, len(dirty_df)),
            random_state=42
        )

        dirty_df = pd.concat(
            [dirty_df, duplicates],
            ignore_index=True
        )

    # Add missing values
    if missing_pct > 0 and len(dirty_df) > 0:
        rng = np.random.default_rng(42)

        total_cells = dirty_df.size
        missing_count = int(
            total_cells * missing_pct / 100
        )

        for _ in range(missing_count):
            row = rng.integers(0, len(dirty_df))
            col = rng.integers(0, len(dirty_df.columns))

            dirty_df.iloc[row, col] = np.nan

    return dirty_df


def detect_missing_values(df: pd.DataFrame) -> dict:
    """Detect missing values and create a text report."""

    null_counts = df.isnull().sum()
    null_percentages = (
        null_counts / len(df)
    ) * 100

    print("\nMissing Value Detection")
    print("=======================")
    print(null_counts)

    print("\nHeatmap-Style Null Report")
    print("=========================")

    for column in df.columns:
        count = null_counts[column]
        percentage = null_percentages[column]

        bar = "#" * int(percentage / 5)

        print(
            f"{column:12} | {bar:<20} "
            f"{count} missing "
            f"({percentage:.1f}%)"
        )

    return {
        "null_counts": null_counts.to_dict(),
        "null_percentages": null_percentages.to_dict()
    }


def main():
    filepath = (
        "day6_series_dataframes_data_ingestion/data/"
        "ecommerce_orders.csv"
    )

    df = pd.read_csv(filepath)

    dirty_df = inject_dirty_data(
        df,
        missing_pct=5,
        dup_count=10
    )

    print("Original Shape:", df.shape)
    print("Dirty Shape:", dirty_df.shape)

    detect_missing_values(dirty_df)

    print(
        "\nDuplicate Rows:",
        dirty_df.duplicated().sum()
    )


if __name__ == "__main__":
    main()