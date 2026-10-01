import warnings

import numpy as np
import pytest

from day19_logistic_regression import sigmoid


def test_sigmoid_zero():
    assert sigmoid(np.array([0.0]))[0] == 0.5


def test_sigmoid_stable_for_large_values():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sigmoid(np.array([-1000.0, 1000.0]))
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(1.0, abs=1e-12)


def test_sigmoid_symmetry_and_range():
    z = np.linspace(-50, 50, 101)
    s = sigmoid(z)
    assert np.allclose(s + sigmoid(-z), 1.0)
    assert np.all((s >= 0) & (s <= 1))
