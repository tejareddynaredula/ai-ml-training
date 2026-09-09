import numpy as np


def filter_outliers(arr: np.ndarray, threshold: float) -> np.ndarray:
    return arr[np.abs(arr) <= threshold]


def extract_submatrix(
    arr: np.ndarray, row_range: tuple, col_range: tuple
) -> np.ndarray:
    return arr[row_range[0] : row_range[1], col_range[0] : col_range[1]]


def normalize_broadcast(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D")

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    if np.any(std == 0):
        raise ValueError("Cannot normalize a column with zero standard deviation")

    # Broadcasting subtracts each column mean from every row.
    # The column standard deviation is also broadcast across all rows.
    return (arr - mean) / std


def select_rows_by_index(arr: np.ndarray, indices: list) -> np.ndarray:
    return arr[np.array(indices)]


def main():
    # Basic 1D slicing
    arr1 = np.arange(1, 11)
    print("1D array:", arr1)
    print("Elements 2 to 5:", arr1[1:5])

    # Basic 2D slicing
    arr2 = np.arange(1, 13).reshape(3, 4)
    print("\n2D array:\n", arr2)
    print("First two rows:\n", arr2[:2, :])
    print("First two columns:\n", arr2[:, :2])
    print("Sub-matrix:\n", arr2[1:3, 1:3])

    # Practical function demonstrations
    print("\nFiltered outliers:", filter_outliers(arr1, 6))
    print("Selected rows:\n", select_rows_by_index(arr2, [0, 2]))

    data = np.array([[10, 100], [20, 200], [30, 300]])
    print("\nBefore normalization:\n", data)

    normalized = normalize_broadcast(data)
    print("After normalization:\n", normalized)
    print("Column means:", normalized.mean(axis=0))
    print("Column std:", normalized.std(axis=0))


if __name__ == "__main__":
    main()