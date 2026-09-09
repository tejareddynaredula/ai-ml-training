import numpy as np
import pytest

from task4_practical_functions import (
    filter_outliers,
    extract_submatrix,
    normalize_broadcast,
    select_rows_by_index,
)


def test_filter_outliers_with_outliers():
    arr = np.array([10, 20, 100])
    assert np.array_equal(filter_outliers(arr, 50), [10, 20])


def test_filter_outliers_no_outliers():
    arr = np.array([10, 20, 30])
    assert np.array_equal(filter_outliers(arr, 50), arr)


def test_filter_outliers_all_outliers():
    arr = np.array([100, 200, 300])
    assert len(filter_outliers(arr, 50)) == 0


def test_extract_submatrix():
    arr = np.arange(1, 17).reshape(4, 4)
    result = extract_submatrix(arr, (1, 3), (1, 3))
    assert np.array_equal(result, [[6, 7], [10, 11]])


def test_select_rows_by_index():
    arr = np.arange(1, 13).reshape(3, 4)
    result = select_rows_by_index(arr, [0, 2])
    assert np.array_equal(result, [[1, 2, 3, 4], [9, 10, 11, 12]])


def test_normalize_mean():
    arr = np.array([[10, 100], [20, 200], [30, 300]])
    result = normalize_broadcast(arr)
    assert np.allclose(result.mean(axis=0), 0)


def test_normalize_std():
    arr = np.array([[10, 100], [20, 200], [30, 300]])
    result = normalize_broadcast(arr)
    assert np.allclose(result.std(axis=0), 1)


def test_normalize_rejects_1d_array():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        normalize_broadcast(arr)