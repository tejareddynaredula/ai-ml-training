import numpy as np
import pytest

from task2_loop_vs_vectorized import (
    sum_of_squares_loop,
    sum_of_squares_vectorized,
    dot_product_loop,
    dot_product_vectorized,
    elementwise_distance_loop,
    elementwise_distance_vectorized,
)

from task6_practical_functions import (
    euclidean_distance_matrix,
    matrix_ops_report,
)


def test_sum_of_squares():
    arr = np.array([1, 2, 3])
    assert sum_of_squares_loop(arr) == sum_of_squares_vectorized(arr)


def test_dot_product():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert dot_product_loop(a, b) == dot_product_vectorized(a, b)


def test_elementwise_distance():
    a = np.array([1, 5, 10])
    b = np.array([2, 3, 7])
    assert np.allclose(
        elementwise_distance_loop(a, b),
        elementwise_distance_vectorized(a, b),
    )


def test_euclidean_distance():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 3], [4, 6]])
    result = euclidean_distance_matrix(a, b)
    assert np.allclose(result, [np.sqrt(2), np.sqrt(5)])


def test_matrix_product():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = matrix_ops_report(a, b)
    assert np.array_equal(result["product"], a @ b)


def test_matrix_inverse():
    a = np.array([[1, 2], [3, 4]])
    b = np.eye(2)
    result = matrix_ops_report(a, b)
    assert np.allclose(result["inverse"], np.linalg.inv(a))


def test_non_invertible_matrix():
    a = np.array([[1, 2], [2, 4]])
    b = np.eye(2)
    result = matrix_ops_report(a, b)
    assert result["inverse"] is None


def test_distance_shape_error():
    a = np.array([[1, 2]])
    b = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        euclidean_distance_matrix(a, b)