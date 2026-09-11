import numpy as np


def rank_students(scores: np.ndarray) -> np.ndarray:
    totals = np.sum(scores, axis=1)
    return np.argsort(totals)[::-1]


def main():
    np.random.seed(42)
    scores = np.random.randint(0, 101, size=(200, 5))

    ranking = rank_students(scores)

    print("Student Ranking")
    print("----------------")
    print("Top 10 students (index):")
    print(ranking[:10])

    totals = np.sum(scores, axis=1)

    print("\nTop 10 total scores:")
    print(totals[ranking[:10]])


if __name__ == "__main__":
    main()