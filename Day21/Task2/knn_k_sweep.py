import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def k_sweep_analysis(X_train, X_test, y_train, y_test):
    """
    Train KNN models for k values from 1 to 20
    and return their accuracy scores.
    """
    results = []

    for k in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        results.append({
            "k": k,
            "accuracy": accuracy
        })

    return pd.DataFrame(results)


def plot_k_sweep(results):
    """Plot K value against test accuracy."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        results["k"],
        results["accuracy"],
        marker="o"
    )

    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Test Accuracy")
    plt.title("KNN Accuracy vs K")
    plt.xticks(range(1, 21))
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Day21/Task2/knn_k_sweep.png", dpi=150)
    plt.show()


def main():
    # Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features because KNN is distance-based
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Run k sweep
    results = k_sweep_analysis(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    print("KNN K-Sweep Analysis")
    print("=" * 50)

    print(results.to_string(index=False))

    # Find best k
    best_row = results.loc[results["accuracy"].idxmax()]

    print("\nBest K:")
    print(f"k = {int(best_row['k'])}")

    print(f"Accuracy = {best_row['accuracy']:.4f}")

    # Plot
    plot_k_sweep(results)


if __name__ == "__main__":
    main()