import warnings

import numpy as np
import pytest

from day19_logistic_regression import sigmoid, logistic_regression_fit


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


def _toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return np.column_stack((np.ones(n), X)), y


def test_fit_loss_decreases_and_history_length():
    X, y = _toy_data()
    _, losses = logistic_regression_fit(X, y, lr=0.1, epochs=200)
    assert len(losses) == 201
    assert losses[-1] < losses[0]


def test_scratch_agrees_with_sklearn():
    from sklearn.linear_model import LogisticRegression
    X, y = _toy_data()
    coef, _ = logistic_regression_fit(X, y, lr=0.5, epochs=3000)
    scratch = sigmoid(X @ coef) >= 0.5
    sk = LogisticRegression(C=1e6, max_iter=1000).fit(X[:, 1:], y).predict(X[:, 1:])
    assert np.mean(scratch == sk.astype(bool)) >= 0.90


def test_fit_rejects_bad_labels():
    X, _ = _toy_data(n=10)
    with pytest.raises(ValueError):
        logistic_regression_fit(X, np.arange(10), lr=0.1, epochs=5)
