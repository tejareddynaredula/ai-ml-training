import numpy as np


def main():
    # Example 1: Scalar + array
    # A scalar is broadcast to every element of the array.
    arr = np.array([1, 2, 3])
    print("Scalar + array:", arr + 10)

    # Example 2: 1D + 2D
    # The 1D array is matched with the columns of the 2D array
    # and broadcast across each row.
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr1d = np.array([10, 20, 30])
    print("1D + 2D:\n", arr2d + arr1d)

    # Example 3: Mismatched shapes
    # Broadcasting works when dimensions are equal or one dimension is 1.
    # Shapes (2, 3) and (2, 2) cannot be broadcast together.
    a = np.ones((2, 3))
    b = np.ones((2, 2))

    try:
        print("Mismatched shapes:\n", a + b)
    except ValueError as e:
        print("Broadcasting error:", e)


if __name__ == "__main__":
    main()