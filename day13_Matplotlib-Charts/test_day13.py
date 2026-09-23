"""
Day 13 - Matplotlib Charts

Tests for Matplotlib visualization functions.
"""

import os

import matplotlib
import pandas as pd
import pytest

from task3_boxplot import plot_boxplot_by_category
from task4_scatter import plot_scatter_correlation
from task5_bar_chart import plot_category_counts
from task6_dashboard import plot_dashboard_grid


matplotlib.use("Agg")


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "Quantity": [1, 2, 3, 4, 5],
        "Unit_Price": [10, 20, 15, 30, 25],
        "Category": [
            "Electronics",
            "Clothing",
            "Electronics",
            "Food",
            "Clothing"
        ]
    })


def test_boxplot_creates_png(sample_dataframe, tmp_path):
    output = tmp_path / "boxplot.png"

    plot_boxplot_by_category(
        sample_dataframe,
        "Unit_Price",
        "Category",
        str(output)
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_boxplot_single_category(tmp_path):
    df = pd.DataFrame({
        "Unit_Price": [10, 20, 30],
        "Category": ["Food", "Food", "Food"]
    })

    output = tmp_path / "single_category_boxplot.png"

    plot_boxplot_by_category(
        df,
        "Unit_Price",
        "Category",
        str(output)
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_boxplot_empty_dataframe_raises_error(tmp_path):
    df = pd.DataFrame({
        "Unit_Price": [],
        "Category": []
    })

    output = tmp_path / "empty_boxplot.png"

    with pytest.raises(ValueError):
        plot_boxplot_by_category(
            df,
            "Unit_Price",
            "Category",
            str(output)
        )


def test_scatter_creates_png(sample_dataframe, tmp_path):
    output = tmp_path / "scatter.png"

    plot_scatter_correlation(
        sample_dataframe,
        "Quantity",
        "Unit_Price",
        str(output)
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_bar_chart_creates_png(sample_dataframe, tmp_path):
    output = tmp_path / "bar_chart.png"

    plot_category_counts(
        sample_dataframe,
        "Category",
        str(output)
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_dashboard_creates_png(sample_dataframe, tmp_path):
    output = tmp_path / "dashboard.png"

    plot_dashboard_grid(
        sample_dataframe,
        str(output)
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_bar_chart_empty_dataframe_raises_error(tmp_path):
    df = pd.DataFrame({
        "Category": []
    })

    output = tmp_path / "empty_bar_chart.png"

    with pytest.raises(ValueError):
        plot_category_counts(
            df,
            "Category",
            str(output)
        )