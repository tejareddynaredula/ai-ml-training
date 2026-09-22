"""
Day 12 - Task 4: Probability Distributions

Generate and analyze:
- Normal distribution
- Binomial distribution
- Poisson distribution
"""

import numpy as np


def generate_normal_distribution(
    mean: float = 100,
    std: float = 15,
    size: int = 1000
) -> np.ndarray:
    """Generate samples from a normal distribution."""

    return np.random.normal(
        loc=mean,
        scale=std,
        size=size
    )


def generate_binomial_distribution(
    trials: int = 10,
    probability: float = 0.5,
    size: int = 1000
) -> np.ndarray:
    """Generate samples from a binomial distribution."""

    return np.random.binomial(
        n=trials,
        p=probability,
        size=size
    )


def generate_poisson_distribution(
    rate: float = 4,
    size: int = 1000
) -> np.ndarray:
    """Generate samples from a Poisson distribution."""

    return np.random.poisson(
        lam=rate,
        size=size
    )


def analyze_distribution(
    name: str,
    data: np.ndarray
) -> None:
    """Print basic statistics for a distribution."""

    print(f"\n{name}")
    print("-" * 50)
    print(f"Sample size: {len(data)}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Standard deviation: {np.std(data):.4f}")
    print(f"Minimum: {np.min(data):.4f}")
    print(f"Maximum: {np.max(data):.4f}")


def main():
    """Generate and analyze all three distributions."""

    # Make results reproducible
    np.random.seed(42)

    normal_data = generate_normal_distribution()

    binomial_data = generate_binomial_distribution()

    poisson_data = generate_poisson_distribution()

    print("TASK 4: PROBABILITY DISTRIBUTIONS")
    print("=" * 50)

    analyze_distribution(
        "Normal Distribution",
        normal_data
    )

    analyze_distribution(
        "Binomial Distribution",
        binomial_data
    )

    analyze_distribution(
        "Poisson Distribution",
        poisson_data
    )


if __name__ == "__main__":
    main()