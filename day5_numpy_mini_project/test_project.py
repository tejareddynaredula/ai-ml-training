import numpy as np
import pytest

from common.numpy_utils import zscore_normalize
from day5_numpy_mini_project import task3_subject_statistics as task3
from day5_numpy_mini_project import task4_student_ranking as task4
from day5_numpy_mini_project import task5_at_risk_detection as task5
from day5_numpy_mini_project import task7_subject_correlation as task7
from day5_numpy_mini_project.task3_subject_statistics import compute_subject_stats
from day5_numpy_mini_project.task4_student_ranking import rank_students
from day5_numpy_mini_project.task5_at_risk_detection import (
    identify_at_risk_students,
)
from day5_numpy_mini_project.task7_subject_correlation import (
    correlation_between_subjects,
)


# Task 3: Subject Statistics Tests

def test_mean():
    scores = np.array([[10, 20], [30, 40]])
    stats = compute_subject_stats(scores)

    assert np.array_equal(stats["mean"], [20, 30])


def test_median():
    scores = np.array([[10, 20], [30, 40]])
    stats = compute_subject_stats(scores)

    assert np.array_equal(stats["median"], [20, 30])


def test_min():
    scores = np.array([[10, 20], [30, 40]])
    stats = compute_subject_stats(scores)

    assert np.array_equal(stats["min"], [10, 20])


def test_max():
    scores = np.array([[10, 20], [30, 40]])
    stats = compute_subject_stats(scores)

    assert np.array_equal(stats["max"], [30, 40])


def test_subject_stats_std():
    scores = np.array([[10, 20], [30, 40]])
    stats = compute_subject_stats(scores)

    assert np.allclose(stats["std"], [10, 10])


# Task 4: Student Ranking Tests

def test_ranking():
    scores = np.array([[50, 50], [90, 90], [60, 60]])
    ranking = rank_students(scores)

    assert np.array_equal(ranking, [1, 2, 0])


def test_top_student():
    scores = np.array([[10, 10], [100, 100]])
    ranking = rank_students(scores)

    assert ranking[0] == 1


def test_ranking_order():
    scores = np.array([[20, 20], [50, 50], [30, 30]])
    ranking = rank_students(scores)

    assert np.array_equal(ranking, [1, 2, 0])


# Task 5: At-Risk Detection Tests

def test_at_risk_detection():
    scores = np.array([[30, 30], [80, 80], [20, 40]])
    result = identify_at_risk_students(scores, 40)

    assert np.array_equal(result, [True, False, True])


def test_no_at_risk_students():
    scores = np.array([[50, 60], [70, 80]])
    result = identify_at_risk_students(scores, 40)

    assert not np.any(result)


def test_at_risk_count():
    scores = np.array([[20, 20], [50, 50], [30, 30]])
    result = identify_at_risk_students(scores, 40)

    assert np.sum(result) == 2


# Task 6: Score Normalization Tests

def test_normalization_mean():
    scores = np.array([[10, 20], [30, 40], [50, 60]])
    normalized = zscore_normalize(scores)

    assert np.allclose(np.mean(normalized, axis=0), 0)


def test_normalization_std():
    scores = np.array([[10, 20], [30, 40], [50, 60]])
    normalized = zscore_normalize(scores)

    assert np.allclose(np.std(normalized, axis=0), 1)


def test_normalization_zero_std():
    scores = np.array([[1, 2], [1, 4]])

    with pytest.raises(ValueError):
        zscore_normalize(scores)


# Task 7: Subject Correlation Tests

def test_correlation_shape():
    scores = np.array([[10, 20], [30, 40], [50, 60]])
    correlation = correlation_between_subjects(scores)

    assert correlation.shape == (2, 2)


def test_correlation_symmetry():
    scores = np.array([[10, 20], [30, 40], [50, 60]])
    correlation = correlation_between_subjects(scores)

    assert np.allclose(correlation, correlation.T)


def test_correlation_values():
    scores = np.array([[10, 20], [20, 40], [30, 60]])
    correlation = correlation_between_subjects(scores)

    assert np.allclose(correlation[0, 1], 1.0)


# Task 10: Main Function Tests for Coverage

def test_task3_main(capsys):
    task3.main()

    output = capsys.readouterr().out

    assert "Subject Statistics" in output


def test_task4_main(capsys):
    task4.main()

    output = capsys.readouterr().out

    assert "Student Ranking" in output


def test_task5_main(capsys):
    task5.main()

    output = capsys.readouterr().out

    assert "At-Risk Student Detection" in output


def test_task7_main(capsys):
    task7.main()

    output = capsys.readouterr().out

    assert "Subject Correlation" in output