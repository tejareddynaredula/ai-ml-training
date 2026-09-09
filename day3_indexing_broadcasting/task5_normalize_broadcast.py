import numpy as np


def normalize_broadcast(arr: np.ndarray) -> np.ndarray:
    """Normalize each column using NumPy broadcasting."""

    if arr.ndim != 2:
        raise ValueError("Input array must be a 2D array")

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    if mean.shape != std.shape:
        raise ValueError("Mean and standard deviation shapes are incompatible")

    if np.any(std == 0):
        raise ValueError("Cannot normalize a column with zero standard deviation")

    # Broadcasting:
    # arr has shape (rows, columns)
    # mean and std have shape (columns,)
    # NumPy broadcasts them across every row.
    return (arr - mean) / std


def main():
    data = np.array(
        [
            [10, 100],
            [20, 200],
            [30, 300],
        ]
    )

    normalized = normalize_broadcast(data)

    print("Before normalization:")
    print(data)

    print("\nAfter normalization:")
    print(normalized)

    print("\nColumn means:", normalized.mean(axis=0))
    print("Column standard deviations:", normalized.std(axis=0))


if __name__ == "__main__":
    main()