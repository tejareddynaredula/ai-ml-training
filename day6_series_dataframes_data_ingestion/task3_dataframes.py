import numpy as np
import pandas as pd


def dataframe_from_dict():
    data = {
        "Name": ["Teja", "Rahul", "Priya"],
        "Age": [23, 24, 22],
        "Score": [85, 90, 88],
    }

    return pd.DataFrame(data)


def dataframe_from_list():
    data = [
        {"Name": "Teja", "Age": 23, "Score": 85},
        {"Name": "Rahul", "Age": 24, "Score": 90},
        {"Name": "Priya", "Age": 22, "Score": 88},
    ]

    return pd.DataFrame(data)


def dataframe_from_numpy():
    data = np.array(
        [
            [1, 85],
            [2, 90],
            [3, 88],
        ]
    )

    return pd.DataFrame(
        data,
        columns=["Student_ID", "Score"],
    )


def main():
    print("DataFrame from Dictionary of Lists")
    print("----------------------------------")
    print(dataframe_from_dict())

    print("\nDataFrame from List of Dictionaries")
    print("-----------------------------------")
    print(dataframe_from_list())

    print("\nDataFrame from NumPy 2D Array")
    print("-----------------------------")
    print(dataframe_from_numpy())


if __name__ == "__main__":
    main()