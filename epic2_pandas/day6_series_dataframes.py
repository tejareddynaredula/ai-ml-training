import numpy as np
import pandas as pd


def demonstrate_series():
    print("Pandas Series Demonstration")
    print("===========================")

    # Series from a list
    series_from_list = pd.Series([10, 20, 30, 40])
    print("\n1. Series from a list:")
    print(series_from_list)

    # Series from a dictionary
    series_from_dict = pd.Series(
        {
            "Math": 85,
            "Physics": 90,
            "Chemistry": 88,
        }
    )
    print("\n2. Series from a dictionary:")
    print(series_from_dict)

    # Series from a NumPy array
    array = np.array([100, 200, 300, 400])
    series_from_array = pd.Series(array)
    print("\n3. Series from a NumPy array:")
    print(series_from_array)

    # Series with custom index labels
    custom_series = pd.Series(
        [95, 88, 92],
        index=["Teja", "Rahul", "Priya"],
    )
    print("\n4. Series with custom index labels:")
    print(custom_series)


if __name__ == "__main__":
    demonstrate_series()