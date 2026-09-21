
"""
Day 11 - Task 7: Unit Tests

Test statistics and probability functions.
"""

import numpy as np
import pandas as pd
import pytest

from day11_descriptive_stats import (
    compute_central_tendency,
    compute_spread,
    conditional_probability,
    bayes_theorem,
)


def test_mean():
    data = np.array([2, 4, 4, 6, 8])
    result = compute_central_tendency(data)
    assert result["mean"] == pytest.approx(4.8)


def test_median():
    data = np.array([2, 4, 4, 6, 8])
    result = compute_central_tendency(data)
    assert result["median"] == pytest.approx(4.0)


def test_mode():
    data = np.array([2, 4, 4, 6, 8])
    result = compute_central_tendency(data)
    assert 4.0 in result["mode"]


def test_variance():
    data = np.array([2, 4, 4, 6, 8])
    result = compute_spread(data)
    assert result["variance"] == pytest.approx(4.16)


def test_standard_deviation():
    data = np.array([2, 4, 4, 6, 8])
    result = compute_spread(data)
    assert result["std"] == pytest.approx(2.0396078)


def test_range():
    data = np.array([2, 4, 4, 6, 8])
    result = compute_spread(data)
    assert result["range"] == pytest.approx(6.0)


def test_iqr():
    data = np.array([2, 4, 4, 6, 8])
    result = compute_spread(data)
    assert result["iqr"] == pytest.approx(2.0)


def test_conditional_probability():
    df = pd.DataFrame({
        "Category": ["Electronics", "Electronics", "Furniture"],
        "Unit_Price": [35000, 20000, 40000]
    })

    result = conditional_probability(
        df,
        "Unit_Price > 30000",
        "Category == 'Electronics'"
    )

    assert result == pytest.approx(0.5)


def test_bayes_theorem():
    result = bayes_theorem(0.30, 0.80, 0.40)
    assert result == pytest.approx(0.60)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])