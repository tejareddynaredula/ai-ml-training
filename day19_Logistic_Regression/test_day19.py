import warnings

import numpy as np
import pytest

from day19_logistic_regression import sigmoid, logistic_regression_fit, classification_report_dict, threshold_analysis


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


Y_TRUE = np.array([0, 0, 1, 1, 1, 0, 1, 0])
Y_PROB = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.6])


def test_report_metrics_in_unit_interval():
    rep = classification_report_dict(Y_TRUE, (Y_PROB >= 0.5).astype(int), Y_PROB)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0.0 <= rep[key] <= 1.0


def test_report_confusion_matrix_counts_all_samples():
    rep = classification_report_dict(Y_TRUE, (Y_PROB >= 0.5).astype(int), Y_PROB)
    assert rep["confusion_matrix"].shape == (2, 2)
    assert rep["confusion_matrix"].sum() == len(Y_TRUE)


def test_report_perfect_predictions():
    rep = classification_report_dict(Y_TRUE, Y_TRUE, Y_TRUE.astype(float))
    assert rep["accuracy"] == rep["f1"] == rep["roc_auc"] == 1.0


def test_report_rejects_length_mismatch():
    with pytest.raises(ValueError):
        classification_report_dict(Y_TRUE, Y_TRUE[:-1], Y_PROB)


def test_threshold_analysis_one_row_per_threshold():
    df = threshold_analysis(Y_TRUE, Y_PROB, [0.3, 0.5, 0.7])
    assert len(df) == 3
    assert list(df["threshold"]) == [0.3, 0.5, 0.7]
    assert {"precision", "recall", "F1"} <= set(df.columns)


def test_higher_threshold_never_raises_recall():
    df = threshold_analysis(Y_TRUE, Y_PROB, [0.3, 0.5, 0.7])
    assert df["recall"].is_monotonic_decreasing


def test_threshold_analysis_rejects_bad_threshold():
    with pytest.raises(ValueError):
        threshold_analysis(Y_TRUE, Y_PROB, [1.5])
