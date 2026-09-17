import pandas as pd

from day8_filter_group_merge import merge_datasets


def create_test_data():
    df1 = pd.DataFrame({
        "Category": ["Electronics", "Furniture", "Clothing"],
        "Sales": [100, 200, 150]
    })

    df2 = pd.DataFrame({
        "Category": ["Electronics", "Furniture", "Books"],
        "Department": ["Technology", "Home", "Education"]
    })

    return df1, df2


def test_inner_join():
    df1, df2 = create_test_data()

    result = merge_datasets(df1, df2, "Category", "inner")

    assert len(result) == 2


def test_left_join():
    df1, df2 = create_test_data()

    result = merge_datasets(df1, df2, "Category", "left")

    assert len(result) == 3


def test_right_join():
    df1, df2 = create_test_data()

    result = merge_datasets(df1, df2, "Category", "right")

    assert len(result) == 3


def test_outer_join():
    df1, df2 = create_test_data()

    result = merge_datasets(df1, df2, "Category", "outer")

    assert len(result) == 4


def test_inner_join_values():
    df1, df2 = create_test_data()

    result = merge_datasets(df1, df2, "Category", "inner")

    assert "Technology" in result["Department"].values
    assert "Home" in result["Department"].values