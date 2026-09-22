"""
Day 12 - Task 9: Unit Tests

Tests for correlation, distributions,
normality, and z-score functionality.
"""

import numpy as np
import pandas as pd
import pytest

from task1_pearson_correlation import pearson_correlation
from task3_spearman_correlation import spearman_correlation
from task4_probability_distributions import (
    generate_normal_distribution,
    generate_binomial_distribution,
    generate_poisson_distribution,
)
from task5_normality_check import normality_check
from task6_zscores_outliers import (
    compute_zscores,
    identify_outliers,
)


def test_pearson_correlation():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    result = pearson_correlation(x, y)

    assert result == pytest.approx(1.0)


def test_pearson_negative_correlation():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 8, 6, 4, 2])

    result = pearson_correlation(x, y)

    assert result == pytest.approx(-1.0)


def test_spearman_correlation():
    x = np.array([10, 20, 30, 40, 50])
    y = np.array([15, 25, 35, 45, 60])

    result = spearman_correlation(x, y)

    assert result == pytest.approx(1.0)


def test_zscores_mean():
    data = np.array([10, 20, 30, 40, 50])

    zscores = compute_zscores(data)

    assert np.mean(zscores) == pytest.approx(0.0)


def test_zscores_standard_deviation():
    data = np.array([10, 20, 30, 40, 50])

    zscores = compute_zscores(data)

    assert np.std(zscores) == pytest.approx(1.0)


def test_zscore_outlier_detection():
    data = np.array([
        10, 11, 12, 12, 13,
        13, 14, 14, 15, 15,
        16, 17, 18, 19, 50
    ])

    outliers = identify_outliers(data, threshold=3)

    assert 50 in outliers


def test_normal_distribution_size():
    np.random.seed(42)

    data = generate_normal_distribution(size=1000)

    assert len(data) == 1000


def test_binomial_distribution_size():
    np.random.seed(42)

    data = generate_binomial_distribution(size=1000)

    assert len(data) == 1000


def test_poisson_distribution_size():
    np.random.seed(42)

    data = generate_poisson_distribution(size=1000)

    assert len(data) == 1000


def test_normality_check():
    np.random.seed(42)

    data = np.random.normal(
        loc=100,
        scale=10,
        size=100
    )

    result = normality_check(data)

    assert 0 <= result["p_value"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])