import numpy as np


def generate_dataset(n_students=200, n_subjects=5):
    np.random.seed(42)
    return np.random.randint(0, 101, size=(n_students, n_subjects))


def main():
    scores = generate_dataset()

    print("Dataset shape:", scores.shape)
    print("First 5 students:")
    print(scores[:5])


if __name__ == "__main__":
    main()