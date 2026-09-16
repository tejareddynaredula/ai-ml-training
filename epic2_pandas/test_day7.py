
import pandas as pd
import numpy as np

from task8_cleaning_pipeline import (
    handle_missing_values,
    remove_duplicates,
    fix_dtypes,
    data_quality_report,
    clean_dataset
)


def sample_dataframe():
    return pd.DataFrame({
        "Order_ID": [1, 2, 2, 3],
        "Customer": ["Teja", "Rahul", None, "Priya"],
        "Quantity": [2, 3, None, 5],
        "Unit_Price": [100, 200, 200, 300],
        "Order_Date": [
            "2026-09-01",
            "2026-09-02",
            "2026-09-02",
            "2026-09-03"
        ]
    })


def test_missing_values_are_filled():
    df = sample_dataframe()

    result = handle_missing_values(
        df,
        {
            "Customer": "mode",
            "Quantity": "median"
        }
    )

    assert result["Customer"].isnull().sum() == 0
    assert result["Quantity"].isnull().sum() == 0


def test_mean_strategy():
    df = pd.DataFrame({"Price": [100, 200, np.nan]})

    result = handle_missing_values(
        df,
        {"Price": "mean"}
    )

    assert result["Price"].isnull().sum() == 0
    assert result["Price"].iloc[2] == 150


def test_median_strategy():
    df = pd.DataFrame({"Quantity": [2, 4, np.nan]})

    result = handle_missing_values(
        df,
        {"Quantity": "median"}
    )

    assert result["Quantity"].isnull().sum() == 0
    assert result["Quantity"].iloc[2] == 3


def test_mode_strategy():
    df = pd.DataFrame({
        "Category": ["A", "A", "B", None]
    })

    result = handle_missing_values(
        df,
        {"Category": "mode"}
    )

    assert result["Category"].isnull().sum() == 0
    assert result["Category"].iloc[3] == "A"


def test_duplicate_removal():
    df = pd.DataFrame({
        "ID": [1, 2, 2, 3]
    })

    result = remove_duplicates(df)

    assert len(result) == 3
    assert result.duplicated().sum() == 0


def test_subset_duplicate_removal():
    df = pd.DataFrame({
        "Order_ID": [1, 1, 2],
        "Customer": ["A", "B", "C"]
    })

    result = remove_duplicates(
        df,
        subset=["Order_ID"]
    )

    assert len(result) == 2


def test_dtype_conversion():
    df = pd.DataFrame({
        "Quantity": ["2", "3", "4"],
        "Order_Date": [
            "2026-09-01",
            "2026-09-02",
            "2026-09-03"
        ]
    })

    result = fix_dtypes(
        df,
        {
            "Quantity": "numeric",
            "Order_Date": "datetime"
        }
    )

    assert pd.api.types.is_numeric_dtype(
        result["Quantity"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["Order_Date"]
    )


def test_empty_dataframe():
    df = pd.DataFrame(
        columns=["Customer", "Quantity"]
    )

    result = data_quality_report(df)

    assert result["shape"] == (0, 2)
    assert result["null_counts"] == 0
    assert result["duplicate_counts"] == 0


def test_full_missing_column():
    df = pd.DataFrame({
        "Customer": [None, None, None],
        "Quantity": [1, 2, 3]
    })

    result = handle_missing_values(
        df,
        {
            "Customer": "mode"
        }
    )

    # A completely missing column cannot be filled using mode
    assert result["Customer"].isnull().sum() == 3


def test_cleaning_pipeline():
    df = sample_dataframe()

    result = clean_dataset(df)

    assert result.isnull().sum().sum() == 0
    assert result.duplicated().sum() == 0


def test_quality_report():
    df = sample_dataframe()

    report = data_quality_report(df)

    assert "shape" in report
    assert "null_counts" in report
    assert "duplicate_counts" in report
    assert "dtypes" in report