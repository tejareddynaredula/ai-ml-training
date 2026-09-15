import pytest
from pathlib import Path

from task6_practical_functions import load_dataset


DATA_FOLDER = Path("day6_series_dataframes_data_ingestion/data")


def test_load_csv():
    df = load_dataset(str(DATA_FOLDER / "ecommerce_orders.csv"))
    assert df.shape == (150, 7)


def test_load_excel():
    df = load_dataset(str(DATA_FOLDER / "ecommerce_orders.xlsx"))
    assert df.shape == (150, 7)


def test_load_json():
    df = load_dataset(str(DATA_FOLDER / "ecommerce_orders.json"))
    assert df.shape == (150, 7)


def test_csv_columns():
    df = load_dataset(str(DATA_FOLDER / "ecommerce_orders.csv"))
    assert "Order_ID" in df.columns
    assert "Customer" in df.columns
    assert "Product" in df.columns


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("missing_file.csv")


def test_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset("data.txt")


def test_no_null_values():
    df = load_dataset(str(DATA_FOLDER / "ecommerce_orders.csv"))
    assert df.isnull().sum().sum() == 0