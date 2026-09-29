import unittest

import pandas as pd

from epic4_ml.day16_ml_basics import (
    load_ml_dataset,
    split_data,
    get_baseline_scores,
    demonstrate_leakage,
)


class TestDay16MLBasics(unittest.TestCase):

    def test_diabetes_dataset_loads(self):
        X, y = load_ml_dataset("diabetes")
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(len(X), 442)

    def test_breast_cancer_dataset_loads(self):
        X, y = load_ml_dataset("breast_cancer")
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], 30)
        self.assertEqual(len(X), 569)

    def test_split_is_80_20(self):
        X, y = load_ml_dataset("diabetes")
        X_train, X_test, y_train, y_test = split_data(X, y)

        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertAlmostEqual(len(X_test) / len(X), 0.2, delta=0.01)
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))

    def test_stratified_class_ratios_match(self):
        X, y = load_ml_dataset("breast_cancer")
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, stratify=True
        )

        original_ratios = y.value_counts(normalize=True).sort_index()
        train_ratios = y_train.value_counts(normalize=True).sort_index()
        test_ratios = y_test.value_counts(normalize=True).sort_index()

        for label in original_ratios.index:
            self.assertLessEqual(
                abs(original_ratios[label] - train_ratios[label]), 0.02
            )
            self.assertLessEqual(
                abs(original_ratios[label] - test_ratios[label]), 0.02
            )

    def test_invalid_test_size_raises_value_error(self):
        X, y = load_ml_dataset("diabetes")

        for invalid_size in (0, 1, -0.1, 1.1):
            with self.subTest(test_size=invalid_size):
                with self.assertRaises(ValueError):
                    split_data(X, y, test_size=invalid_size)

    def test_empty_data_raises_value_error(self):
        X = pd.DataFrame()
        y = pd.Series(dtype=float)

        with self.assertRaises(ValueError):
            split_data(X, y)

    def test_regression_baseline_has_expected_keys(self):
        X, y = load_ml_dataset("diabetes")
        X_train, X_test, y_train, y_test = split_data(X, y)

        scores = get_baseline_scores(
            X_train, X_test, y_train, y_test, "regression"
        )

        self.assertIn("model", scores)
        self.assertIn("MAE", scores)
        self.assertIn("RMSE", scores)
        self.assertIn("R2", scores)

    def test_classification_baseline_has_expected_keys(self):
        X, y = load_ml_dataset("breast_cancer")
        X_train, X_test, y_train, y_test = split_data(
            X, y, stratify=True
        )

        scores = get_baseline_scores(
            X_train, X_test, y_train, y_test, "classification"
        )

        self.assertIn("model", scores)
        self.assertIn("accuracy", scores)

    def test_invalid_task_raises_value_error(self):
        X, y = load_ml_dataset("diabetes")
        X_train, X_test, y_train, y_test = split_data(X, y)

        with self.assertRaises(ValueError):
            get_baseline_scores(
                X_train, X_test, y_train, y_test, "invalid"
            )

    def test_leakage_demo_returns_expected_information(self):
        X, y = load_ml_dataset("diabetes")
        result = demonstrate_leakage(X, y)

        self.assertIn("max_test_value_difference", result)
        self.assertIn("difference_measurable", result)
        self.assertGreater(result["max_test_value_difference"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
