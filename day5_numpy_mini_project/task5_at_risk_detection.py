import numpy as np


def identify_at_risk_students(
    scores: np.ndarray, threshold: float
) -> np.ndarray:
    averages = np.mean(scores, axis=1)
    return averages < threshold


def main():
    np.random.seed(42)
    scores = np.random.randint(0, 101, size=(200, 5))

    threshold = 40
    at_risk = identify_at_risk_students(scores, threshold)

    print("At-Risk Student Detection")
    print("-------------------------")
    print("Threshold:", threshold)
    print("Number of at-risk students:", np.sum(at_risk))
    print("At-risk student indices:")
    print(np.where(at_risk)[0])


if __name__ == "__main__":
    main()