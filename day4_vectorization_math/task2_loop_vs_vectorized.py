import numpy as np


# 1. Sum of squares
def sum_of_squares_loop(arr):
    total = 0
    for x in arr:
        total += x ** 2
    return total


def sum_of_squares_vectorized(arr):
    return np.sum(arr ** 2)


# 2. Dot product
def dot_product_loop(a, b):
    total = 0
    for x, y in zip(a, b):
        total += x * y
    return total


def dot_product_vectorized(a, b):
    return np.dot(a, b)


# 3. Element-wise distance
def elementwise_distance_loop(a, b):
    result = []
    for x, y in zip(a, b):
        result.append(abs(x - y))
    return np.array(result)


def elementwise_distance_vectorized(a, b):
    return np.abs(a - b)


def main():
    a = np.array([1, 2, 3, 4])
    b = np.array([4, 5, 6, 7])

    print("Sum of squares:")
    print("Loop:", sum_of_squares_loop(a))
    print("Vectorized:", sum_of_squares_vectorized(a))

    print("\nDot product:")
    print("Loop:", dot_product_loop(a, b))
    print("Vectorized:", dot_product_vectorized(a, b))

    print("\nElement-wise distance:")
    print("Loop:", elementwise_distance_loop(a, b))
    print("Vectorized:", elementwise_distance_vectorized(a, b))


if __name__ == "__main__":
    main()