import numpy as np


def compute_subject_stats(scores: np.ndarray) -> dict:
    return {
        "mean": np.mean(scores, axis=0),
        "median": np.median(scores, axis=0),
        "std": np.std(scores, axis=0),
        "min": np.min(scores, axis=0),
        "max": np.max(scores, axis=0),
    }


def main():
    np.random.seed(42)
    scores = np.random.randint(0, 101, size=(200, 5))

    stats = compute_subject_stats(scores)

    print("Subject Statistics")
    print("------------------")
    print("Mean:", stats["mean"])
    print("Median:", stats["median"])
    print("Standard Deviation:", stats["std"])
    print("Minimum:", stats["min"])
    print("Maximum:", stats["max"])


if __name__ == "__main__":
    main()