import numpy as np


def main():
    # 1D slicing
    arr1 = np.arange(1, 11)
    print("1D array:", arr1)
    print("Elements 2 to 5:", arr1[1:5])

    # 2D slicing
    arr2 = np.arange(1, 13).reshape(3, 4)
    print("\n2D array:\n", arr2)

    # Select rows
    print("First two rows:\n", arr2[:2, :])

    # Select columns
    print("First two columns:\n", arr2[:, :2])

    # Select sub-matrix
    print("Sub-matrix:\n", arr2[1:3, 1:3])


if __name__ == "__main__":
    main()