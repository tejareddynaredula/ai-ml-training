import numpy as np
import pytest

from day2_numpy_basics import (
    create_identity_matrix,
    random_matrix_stats,
    reshape_pipeline,
    stack_arrays,
)


def test_identity_1():
    assert np.array_equal(create_identity_matrix(1), [[1]])


def test_identity_5():
    assert np.array_equal(create_identity_matrix(5), np.eye(5))


def test_valid_reshape():
    arr = np.arange(6)
    assert reshape_pipeline(arr, (2, 3)).shape == (2, 3)


def test_invalid_reshape():
    with pytest.raises(ValueError):
        reshape_pipeline(np.arange(6), (4, 4))


def test_random_stats():
    stats = random_matrix_stats(2, 3)
    assert set(stats) == {"min", "max", "mean", "std"}


def test_stack_arrays():
    arrays = [np.array([1, 2]), np.array([3, 4])]
    assert np.array_equal(stack_arrays(arrays, 0), [[1, 2], [3, 4]])


def test_identity_shape():
    assert create_identity_matrix(3).shape == (3, 3)