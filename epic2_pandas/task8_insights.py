
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


def generate_insights(df: pd.DataFrame) -> list:
    """Generate plain-English insights from the dataset."""
    insights = []

    # Insight 1: Dataset size
    insights.append(
        f"The dataset contains {len(df)} records "
        f"and {len(df.columns)} columns."
    )

    # Insight 2: Most frequently sold product
    most_sold_product = df["Product"].value_counts().idxmax()
    most_sold_count = df["Product"].value_counts().max()

    insights.append(
        f"{most_sold_product} is the most frequently occurring "
        f"product, with {most_sold_count} records."
    )

    # Insight 3: Category distribution
    category_counts = df["Category"].value_counts()
    top_category = category_counts.idxmax()
    top_category_count = category_counts.max()

    insights.append(
        f"{top_category} is the most common category, "
        f"with {top_category_count} records."
    )

    # Insight 4: Average quantity
    average_quantity = df["Quantity"].mean()

    insights.append(
        f"The average order quantity is "
        f"{average_quantity:.2f} units."
    )

    # Insight 5: Correlation
    correlation = df["Quantity"].corr(df["Unit_Price"])

    insights.append(
        f"The correlation between Quantity and Unit_Price "
        f"is {correlation:.4f}, indicating a weak relationship."
    )

    return insights


if __name__ == "__main__":
    df = load_dataset()

    insights = generate_insights(df)

    print("\nEDA Insights:\n")

    for number, insight in enumerate(insights, start=1):
        print(f"{number}. {insight}")