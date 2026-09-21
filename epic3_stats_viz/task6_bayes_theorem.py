
"""
Day 11 - Task 6: Bayes' Theorem

Calculate the probability of spam given a word in an email.
"""


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

    probabilities = [p_a, p_b_given_a, p_b]

    if not all(0 <= value <= 1 for value in probabilities):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    if p_b == 0:
        raise ValueError(
            "P(B) cannot be zero."
        )

    return (p_b_given_a * p_a) / p_b


def main() -> None:
    """Run a worked Bayes' theorem example."""

    # Example:
    # P(Spam) = 0.30
    # P(Word | Spam) = 0.80
    # P(Word) = 0.40

    p_spam = 0.30
    p_word_given_spam = 0.80
    p_word = 0.40

    result = bayes_theorem(
        p_spam,
        p_word_given_spam,
        p_word
    )

    print("TASK 6: BAYES' THEOREM")
    print("-" * 50)
    print(f"P(Spam): {p_spam}")
    print(f"P(Word | Spam): {p_word_given_spam}")
    print(f"P(Word): {p_word}")
    print(f"P(Spam | Word): {result:.4f}")
    print(f"Probability: {result:.2%}")

    print(
        "\nInterpretation: The probability that an email "
        "is spam given that it contains the word is 60%."
    )


if __name__ == "__main__":
    main()