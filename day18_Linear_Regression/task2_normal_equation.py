import numpy as np


def normal_equation_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit linear regression using the normal equation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix. Include a column of ones if an intercept is needed.
    y : np.ndarray
        Target values.

    Returns
    -------
    np.ndarray
        Estimated coefficients, including the intercept if X contains
        a column of ones.

    Notes
    -----
    If X.T @ X is singular, its inverse does not exist. This can happen
    when features are duplicated or linearly dependent. In that case,
    use the pseudoinverse to calculate a stable least-squares solution.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a one- or two-dimensional array.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    if X.shape[0] == 0:
        raise ValueError("X and y cannot be empty.")

    # Normal equation: w = (X.T @ X)^(-1) @ X.T @ y
    xtx = X.T @ X
    xty = X.T @ y

    try:
        coefficients = np.linalg.inv(xtx) @ xty
    except np.linalg.LinAlgError:
        # Handles singular matrices, such as duplicated features.
        coefficients = np.linalg.pinv(xtx) @ xty

    return coefficients
