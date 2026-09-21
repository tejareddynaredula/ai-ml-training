
"""
Day 11: Descriptive Statistics & Probability Fundamentals

Epic 3 - Statistics and Visualization
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


# ---------------------------------------------------------
# 1. CENTRAL TENDENCY
# ---------------------------------------------------------

def compute_central_tendency(data: np.ndarray) -> dict:
    """Calculate mean, median, and mode from scratch."""

    data = np.asarray(data, dtype=float)

    mean_value = np.sum(data) / len(data)

    sorted_data = np.sort(data)
    n = len(sorted_data)

    if n % 2 == 0:
        median_value = (
            sorted_data[n // 2 - 1] + sorted_data[n // 2]
        ) / 2
    else:
        median_value = sorted_data[n // 2]

    unique_values, counts = np.unique(data, return_counts=True)
    max_count = np.max(counts)

    mode_values = unique_values[counts == max_count]

    return {
        "mean": float(mean_value),
        "median": float(median_value),
        "mode": mode_values.tolist(),
    }


# ---------------------------------------------------------
# 2. SPREAD
# ---------------------------------------------------------

def compute_spread(data: np.ndarray) -> dict:
    """
    Calculate variance, standard deviation, range, and IQR.

    Population variance/std use ddof=0.
    Sample variance/std use ddof=1.

    ddof=0:
        Use when the data represents the entire population.

    ddof=1:
        Use when the data is a sample of a larger population.
        This applies Bessel's correction.
    """

    data = np.asarray(data, dtype=float)

    # Population variance and standard deviation
    mean_value = np.sum(data) / len(data)
    population_variance = np.sum(
        (data - mean_value) ** 2
    ) / len(data)

    population_std = np.sqrt(population_variance)

    # Range
    data_range = np.max(data) - np.min(data)

    # IQR = Q3 - Q1
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    return {
        "variance": float(population_variance),
        "std": float(population_std),
        "range": float(data_range),
        "iqr": float(iqr),
    }


# ---------------------------------------------------------
# 3. CONDITIONAL PROBABILITY
# ---------------------------------------------------------

def conditional_probability(
    df: pd.DataFrame,
    event_a: str,
    event_b: str
) -> float:
    """
    Calculate P(A | B).

    event_a and event_b are pandas query conditions.

    Formula:
        P(A | B) = P(A and B) / P(B)
    """

    event_b_df = df.query(event_b)

    if len(event_b_df) == 0:
        raise ValueError(
            "Event B has zero occurrences."
        )

    both_events_df = event_b_df.query(event_a)

    return len(both_events_df) / len(event_b_df)


# ---------------------------------------------------------
# 4. BAYES' THEOREM
# ---------------------------------------------------------

def bayes_theorem(
    p_a: float,
    p_b_given_a: float,
    p_b: float
) -> float:
    """
    Calculate P(A | B) using Bayes' theorem.

    Formula:
        P(A | B) = P(B | A) * P(A) / P(B)
    """

    if p_b == 0:
        raise ValueError(
            "P(B) cannot be zero."
        )

    if not all(
        0 <= value <= 1
        for value in [p_a, p_b_given_a, p_b]
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    return (p_b_given_a * p_a) / p_b


# ---------------------------------------------------------
# 5. LIBRARY COMPARISON
# ---------------------------------------------------------

def compare_with_libraries(data: np.ndarray) -> None:
    """Compare our calculations with NumPy, Pandas, and SciPy."""

    data = np.asarray(data, dtype=float)

    our_central = compute_central_tendency(data)
    our_spread = compute_spread(data)

    library_mean = np.mean(data)
    library_median = np.median(data)
    library_variance = np.var(data, ddof=0)
    library_std = np.std(data, ddof=0)
    library_range = np.ptp(data)
    library_iqr = stats.iqr(data)

    print("\nSTATISTICS COMPARISON")
    print("-" * 70)
    print(f"{'Statistic':<20}{'From Scratch':<20}{'Library':<20}")
    print("-" * 70)

    comparisons = [
        ("Mean", our_central["mean"], library_mean),
        ("Median", our_central["median"], library_median),
        ("Variance", our_spread["variance"], library_variance),
        ("Standard Deviation", our_spread["std"], library_std),
        ("Range", our_spread["range"], library_range),
        ("IQR", our_spread["iqr"], library_iqr),
    ]

    for name, scratch_value, library_value in comparisons:
        print(
            f"{name:<20}"
            f"{scratch_value:<20.6f}"
            f"{library_value:<20.6f}"
        )

        assert np.isclose(
            scratch_value,
            library_value,
            atol=1e-6
        )

    print("\nAll statistics match within tolerance.")


# ---------------------------------------------------------
# 6. BAYES EXAMPLE
# ---------------------------------------------------------

def run_bayes_example() -> None:
    """
    Worked example:

    P(Spam) = 0.30
    P(Spam word | Spam) = 0.80
    P(Spam word) = 0.40

    Find:
        P(Spam | Spam word)
    """

    p_spam = 0.30
    p_word_given_spam = 0.80
    p_word = 0.40

    result = bayes_theorem(
        p_spam,
        p_word_given_spam,
        p_word
    )

    print("\nBAYES THEOREM EXAMPLE")
    print("-" * 40)
    print(f"P(Spam): {p_spam}")
    print(f"P(Word | Spam): {p_word_given_spam}")
    print(f"P(Word): {p_word}")
    print(f"P(Spam | Word): {result:.4f}")

    print(
        "Interpretation: The probability that an email is spam "
        "given that it contains the word is "
        f"{result:.2%}."
    )


# ---------------------------------------------------------
# 7. DATASET EXECUTION
# ---------------------------------------------------------

def find_dataset() -> Path:
    """Find the Epic 2 cleaned dataset."""

    possible_paths = [
        Path(
            "day9_Exploratory Data Analysis (EDA) Deliverable"
        ) / "cleaned_dataset.csv",

        Path(
            "day9_Exploratory Data Analysis (EDA) Deliverable"
        ) / "data" / "cleaned_dataset.csv",

        Path(
            "epic2_pandas"
        ) / "data" / "cleaned_dataset.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find cleaned_dataset.csv. "
        "Please check the dataset path."
    )


def main() -> None:
    """Run Day 11 statistics and probability exercises."""

    dataset_path = find_dataset()
    df = pd.read_csv(dataset_path)

    print("DAY 11: DESCRIPTIVE STATISTICS")
    print("=" * 50)
    print(f"Dataset loaded from: {dataset_path}")
    print(f"Dataset shape: {df.shape}")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        print(f"\nColumn: {column}")
        compare_with_libraries(
            df[column].dropna().to_numpy()
        )

    # Conditional probability example
    if "Category" in df.columns and "Unit_Price" in df.columns:
        probability = conditional_probability(
            df,
            "Unit_Price > 30000",
            "Category == 'Electronics'"
        )

        print("\nCONDITIONAL PROBABILITY")
        print("-" * 40)
        print(
            "P(Unit_Price > 30000 | "
            f"Category == Electronics): {probability:.4f}"
        )

    run_bayes_example()


if __name__ == "__main__":
     main()