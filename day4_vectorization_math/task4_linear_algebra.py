import numpy as np


def matrix_operations(a, b):
    result = {
        "dot": np.dot(a, b),
        "matmul": a @ b,
        "transpose_a": a.T,
    }

    if a.shape[0] == a.shape[1]:
        result["determinant"] = np.linalg.det(a)

        try:
            result["inverse"] = np.linalg.inv(a)
        except np.linalg.LinAlgError:
            result["inverse"] = None

    return result


def main():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    result = matrix_operations(a, b)

    print("Dot product:\n", result["dot"])
    print("\nMatrix multiplication (@):\n", result["matmul"])
    print("\nTranspose:\n", result["transpose_a"])
    print("\nDeterminant:", result["determinant"])
    print("\nInverse:\n", result["inverse"])

    # Non-invertible matrix
    singular = np.array([[1, 2], [2, 4]])

    try:
        np.linalg.inv(singular)
    except np.linalg.LinAlgError:
        print("\nNon-invertible matrix: handled successfully")


if __name__ == "__main__":
    main()