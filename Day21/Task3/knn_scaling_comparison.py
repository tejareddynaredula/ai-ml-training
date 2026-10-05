import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def compare_scaling_effect(X_train, X_test, y_train, y_test, k=5):
    """
    Compare KNN performance with and without feature scaling.

    Returns
    -------
    pd.DataFrame
        Accuracy comparison.
    """
    # KNN without scaling
    knn_unscaled = KNeighborsClassifier(n_neighbors=k)

    knn_unscaled.fit(X_train, y_train)

    unscaled_predictions = knn_unscaled.predict(X_test)

    unscaled_accuracy = accuracy_score(
        y_test,
        unscaled_predictions
    )

    # StandardScaler
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # KNN with scaling
    knn_scaled = KNeighborsClassifier(n_neighbors=k)

    knn_scaled.fit(X_train_scaled, y_train)

    scaled_predictions = knn_scaled.predict(X_test_scaled)

    scaled_accuracy = accuracy_score(
        y_test,
        scaled_predictions
    )

    results = pd.DataFrame({
        "method": [
            "Without Scaling",
            "StandardScaler"
        ],
        "accuracy": [
            unscaled_accuracy,
            scaled_accuracy
        ]
    })

    return results


def plot_scaling_comparison(results):
    """Plot KNN accuracy with and without scaling."""
    plt.figure(figsize=(8, 5))

    plt.bar(
        results["method"],
        results["accuracy"]
    )

    plt.xlabel("Scaling Method")
    plt.ylabel("Accuracy")
    plt.title("KNN: Effect of Feature Scaling")

    plt.ylim(0, 1.05)
    plt.tight_layout()

    plt.savefig(
        "Day21/Task3/knn_scaling_comparison.png",
        dpi=150
    )

    plt.show()


def main():
    # Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Same split for both experiments
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    k = 5

    results = compare_scaling_effect(
        X_train,
        X_test,
        y_train,
        y_test,
        k=k
    )

    print("KNN Scaling Comparison")
    print("=" * 50)

    print(results.to_string(index=False))

    print("\nAccuracy Difference:")

    difference = (
        results.loc[1, "accuracy"]
        - results.loc[0, "accuracy"]
    )

    print(f"{difference:.4f}")

    # Plot comparison
    plot_scaling_comparison(results)


if __name__ == "__main__":
    main()