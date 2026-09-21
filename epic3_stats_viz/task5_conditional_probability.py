
"""
Day 11 - Task 5: Conditional Probability

Calculate P(A | B) using the Epic 2 dataset.
"""

import pandas as pd
from pathlib import Path


def conditional_probability(
    df: pd.DataFrame,
    event_a: str,
    event_b: str
) -> float:
    """
    Calculate conditional probability P(A | B).

    Formula:
        P(A | B) = P(A and B) / P(B)

    event_a and event_b use pandas query conditions.
    """

    # Filter rows where Event B occurs
    event_b_df = df.query(event_b)

    if len(event_b_df) == 0:
        raise ValueError(
            "Event B has zero occurrences."
        )

    # Filter rows where both A and B occur
    both_events_df = event_b_df.query(event_a)

    probability = (
        len(both_events_df) / len(event_b_df)
    )

    return probability


def find_dataset() -> Path:
    """Find the Epic 2 cleaned dataset."""

    dataset_path = Path(
        "day9_Exploratory Data Analysis (EDA) Deliverable"
    ) / "data" / "cleaned_dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset not found."
        )

    return dataset_path


def main() -> None:
    """Run the conditional probability example."""

    dataset_path = find_dataset()
    df = pd.read_csv(dataset_path)

    # Event A: Unit price is greater than 30,000
    # Event B: Category is Electronics
    event_a = "Unit_Price > 30000"
    event_b = "Category == 'Electronics'"

    probability = conditional_probability(
        df,
        event_a,
        event_b
    )

    print("TASK 5: CONDITIONAL PROBABILITY")
    print("-" * 50)
    print(f"Total records: {len(df)}")
    print(
        "Event B: Category == Electronics"
    )
    print(
        "Event A: Unit_Price > 30000"
    )
    print(
        f"P(A | B): {probability:.4f}"
    )
    print(
        f"Probability: {probability:.2%}"
    )

    print(
        "\nInterpretation: The probability that an order "
        "has a unit price above 30,000 given that it "
        "belongs to Electronics."
    )


if __name__ == "__main__":
    main()