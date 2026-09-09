import numpy as np


def filter_outliers(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Return elements whose absolute value is within the threshold."""
    return arr[np.abs(arr) <= threshold]


def extract_submatrix(
    arr: np.ndarray, row_range: tuple, col_range: tuple
) -> np.ndarray:
    """Extract a submatrix using row and column ranges."""
    return arr[row_range[0] : row_range[1], col_range[0] : col_range[1]]


def normalize_broadcast(arr: np.ndarray) -> np.ndarray:
    """Normalize each column using broadcasting."""
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D")

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    if np.any(std == 0):
        raise ValueError("Cannot normalize a column with zero standard deviation")

    return (arr - mean) / std


def select_rows_by_index(arr: np.ndarray, indices: list) -> np.ndarray:
    """Select rows using fancy indexing."""
    return arr[np.array(indices)]


def main():
    # Filter outliers using boolean masking
    arr = np.array([10, 20, 100, 30, 200])
    print("Filtered:", filter_outliers(arr, 50))

    # Extract a submatrix using slicing
    matrix = np.arange(1, 17).reshape(4, 4)
    print("\nSubmatrix:\n", extract_submatrix(matrix, (1, 3), (1, 3)))

    # Normalize columns using broadcasting
    data = np.array([[1, 10], [2, 20], [3, 30]])
    print("\nNormalized:\n", normalize_broadcast(data))

    # Select rows using fancy indexing
    print("\nSelected rows:\n", select_rows_by_index(matrix, [0, 2]))


if __name__ == "__main__":
    main()