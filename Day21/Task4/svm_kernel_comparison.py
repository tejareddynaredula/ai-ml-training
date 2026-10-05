import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def compare_kernels(X_train, X_test, y_train, y_test):
    """
    Compare SVM linear, RBF, and polynomial kernels.

    Returns
    -------
    pd.DataFrame
        Accuracy, F1-score, and training time for each kernel.
    """
    kernels = ["linear", "rbf", "poly"]
    results = []

    for kernel in kernels:
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel, random_state=42)
        )

        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        results.append({
            "kernel": kernel,
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time_seconds": training_time
        })

    return pd.DataFrame(results)


def plot_svm_decision_boundary(
    X,
    y,
    model,
    ax,
    title,
    feature_names
):
    """
    Plot a 2-feature SVM decision boundary.
    """
    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    predictions = model.predict(grid)
    predictions = predictions.reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        predictions,
        alpha=0.25
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="black"
    )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)


def main():
    # Load Iris dataset
    iris = load_iris()

    # Use only 2 features for visualization
    X = iris.data[:, :2]
    y = iris.target

    feature_names = [
        iris.feature_names[0],
        iris.feature_names[1]
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Compare kernels
    results = compare_kernels(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("SVM Kernel Comparison")
    print("=" * 70)
    print(results.to_string(index=False))

    # Train models for decision-boundary plots
    kernels = ["linear", "rbf", "poly"]

    models = {}

    for kernel in kernels:
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel, random_state=42)
        )

        model.fit(X_train, y_train)
        models[kernel] = model

    # Create side-by-side decision boundary plots
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    for ax, kernel in zip(axes, kernels):
        plot_svm_decision_boundary(
            X,
            y,
            models[kernel],
            ax,
            f"SVM - {kernel.upper()} Kernel",
            feature_names
        )

    fig.suptitle("SVM Kernel Decision Boundaries")

    plt.tight_layout()

    plt.savefig(
        "Day21/Task4/svm_kernel_decision_boundaries.png",
        dpi=150
    )

    plt.show()

    # Save kernel comparison table
    results.to_csv(
        "Day21/Task4/svm_kernel_comparison.csv",
        index=False
    )


if __name__ == "__main__":
    main()