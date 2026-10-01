"""
Day 19 - Logistic Regression from Scratch
"""

from pathlib import Path

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Calculate sigmoid probabilities without overflow warnings."""
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)

    # Separate positive and negative values for numerical stability.
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)

    return result
