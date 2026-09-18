
import pandas as pd

from day9_eda import univariate_report
from task4_outlier_detection import detect_outliers_iqr
from task5_pivot_tables import build_pivot_summary
from task6_transformations import categorize_column
from task9_final_report import generate_final_report


def sample_data():
    return pd.DataFrame({
        "Quantity": [1, 2, 3, 4, 5],
        "Unit_Price": [100, 200, 300, 400, 500],
        "Product": ["A", "B", "A", "B", "A"],
        "Category": [
            "Electronics",
            "Accessories",
            "Electronics",
            "Accessories",
            "Electronics"
        ]
    })


def test_univariate_report_returns_dict():
    df = sample_data()
    result = univariate_report(df)
    assert isinstance(result, dict)


def test_univariate_report_contains_quantity():
    df = sample_data()
    result = univariate_report(df)
    assert "Quantity" in result


def test_outlier_detection_returns_dataframe():
    df = sample_data()
    result = detect_outliers_iqr(df, "Quantity")
    assert isinstance(result, pd.DataFrame)


def test_outlier_detection_detects_outlier():
    df = pd.DataFrame({
        "Quantity": [1, 2, 2, 2, 100]
    })
    result = detect_outliers_iqr(df, "Quantity")
    assert len(result) == 2


def test_pivot_table_returns_dataframe():
    df = sample_data()
    result = build_pivot_summary(
        df,
        "Category",
        "Product",
        "Quantity",
        "sum"
    )
    assert isinstance(result, pd.DataFrame)


def test_categorize_column_creates_new_column():
    df = sample_data()
    result = categorize_column(
        df,
        "Quantity",
        [0, 2, 4, 6],
        ["Low", "Medium", "High"]
    )
    assert "Quantity_Category" in result.columns


def test_final_report_returns_dict():
    df = sample_data()
    result = generate_final_report(df)
    assert isinstance(result, dict)


def test_final_report_contains_insights():
    df = sample_data()
    result = generate_final_report(df)
    assert "insights" in result
    assert len(result["insights"]) == 5