import numpy as np


def correlation_between_subjects(scores: np.ndarray) -> np.ndarray:
    return np.corrcoef(scores, rowvar=False)


def main():
    np.random.seed(42)
    scores = np.random.randint(0, 101, size=(200, 5))

    correlation = correlation_between_subjects(scores)

    print("Subject Correlation")
    print("-------------------")
    print("Correlation matrix shape:", correlation.shape)
    print("\nCorrelation matrix:")
    print(np.round(correlation, 3))


if __name__ == "__main__":
    main()