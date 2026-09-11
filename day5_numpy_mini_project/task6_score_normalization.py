import numpy as np
from common.numpy_utils import zscore_normalize


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    return zscore_normalize(scores)


def main():
    np.random.seed(42)
    scores = np.random.randint(0, 101, size=(200, 5))

    normalized = normalize_scores(scores)

    print("Score Normalization")
    print("-------------------")
    print("Original shape:", scores.shape)
    print("Normalized shape:", normalized.shape)

    print("\nFirst 5 normalized students:")
    print(normalized[:5])

    print("\nNormalized column means:")
    print(np.mean(normalized, axis=0))

    print("\nNormalized column standard deviations:")
    print(np.std(normalized, axis=0))


if __name__ == "__main__":
    main()