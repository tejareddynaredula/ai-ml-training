import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest
import seaborn as sns

from day14_seaborn_advanced import (
    plot_correlation_heatmap,
    plot_pairplot,
    plot_violin_by_category,
    plot_regression_scatter,
)


DATASET_PATH = (
    "day9_Exploratory Data Analysis (EDA) Deliverable"
    "\\data\\cleaned_dataset.csv"
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Order_ID": [1, 2, 3, 4],
        "Quantity": [1, 2, 3, 4],
        "Unit_Price": [10, 20, 30, 40],
        "Category": ["A", "A", "B", "B"],
    })


@pytest.fixture
def real_df():
    return pd.read_csv(DATASET_PATH)


def test_heatmap_creates_file(sample_df, tmp_path):
    output = tmp_path / "heatmap.png"

    plot_correlation_heatmap(
        sample_df,
        str(output)
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_heatmap_correlation_values(sample_df, monkeypatch, tmp_path):
    captured = {}

    def fake_heatmap(data, *args, **kwargs):
        captured["matrix"] = data.copy()

    monkeypatch.setattr(sns, "heatmap", fake_heatmap)

    output = tmp_path / "heatmap.png"

    plot_correlation_heatmap(
        sample_df,
        str(output)
    )

    expected = sample_df.corr(numeric_only=True)

    pd.testing.assert_frame_equal(
        captured["matrix"],
        expected
    )


def test_pairplot_creates_file(real_df, tmp_path):
    output = tmp_path / "pairplot.png"

    plot_pairplot(
        real_df,
        "Category",
        str(output)
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_violin_creates_file(real_df, tmp_path):
    output = tmp_path / "violin.png"

    plot_violin_by_category(
        real_df,
        "Unit_Price",
        "Category",
        str(output)
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_regression_creates_file(real_df, tmp_path):
    output = tmp_path / "regression.png"

    plot_regression_scatter(
        real_df,
        "Quantity",
        "Unit_Price",
        str(output)
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_heatmap_rejects_no_numeric_columns(tmp_path):
    df = pd.DataFrame({
        "Category": ["A", "B", "C"]
    })

    output = tmp_path / "heatmap.png"

    with pytest.raises(ValueError):
        plot_correlation_heatmap(
            df,
            str(output)
        )


def test_pairplot_rejects_invalid_hue(real_df, tmp_path):
    output = tmp_path / "pairplot.png"

    with pytest.raises(ValueError):
        plot_pairplot(
            real_df,
            "Invalid_Category",
            str(output)
        )


def test_violin_rejects_invalid_column(real_df, tmp_path):
    output = tmp_path / "violin.png"

    with pytest.raises(ValueError):
        plot_violin_by_category(
            real_df,
            "Invalid_Column",
            "Category",
            str(output)
        )