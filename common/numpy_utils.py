import numpy as np


def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    if np.any(std == 0):
        raise ValueError("Cannot normalize a column with zero standard deviation")

    return (arr - mean) / std