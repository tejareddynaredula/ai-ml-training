import time

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def tune_rbf_svm(X_train, X_test, y_train, y_test):
    """
    Manually tune C and gamma for an RBF SVM.

    Returns
    -------
    pd.DataFrame
        Results for every C/gamma combination.
    """
    c_values = [0.1, 1, 10, 100]
    gamma_values = ["scale", 0.01, 0.1, 1]

    results = []

    for c in c_values:
        for gamma in gamma_values:
            model = make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=c,
                    gamma=gamma,
                    random_state=42
                )
            )

            start_time = time.perf_counter()

            model.fit(X_train, y_train)

            training_time = time.perf_counter() - start_time

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted"
            )

            results.append({
                "C": c,
                "gamma": gamma,
                "accuracy": accuracy,
                "f1_score": f1,
                "training_time_seconds": training_time
            })

    return pd.DataFrame(results)


def plot_tuning_results(results):
    """Plot accuracy for each C/gamma combination."""
    plot_data = results.copy()

    plot_data["configuration"] = (
        "C=" + plot_data["C"].astype(str)
        + ", gamma=" + plot_data["gamma"].astype(str)
    )

    plt.figure(figsize=(14, 6))

    plt.bar(
        plot_data["configuration"],
        plot_data["accuracy"]
    )

    plt.xlabel("C and Gamma")
    plt.ylabel("Accuracy")
    plt.title("RBF SVM Hyperparameter Tuning")
    plt.xticks(rotation=90)

    plt.ylim(0, 1.05)
    plt.tight_layout()

    plt.savefig(
        "Day21/Task5/svm_rbf_tuning.png",
        dpi=150
    )

    plt.show()


def main():
    # Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Run manual grid search
    results = tune_rbf_svm(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("RBF SVM Hyperparameter Tuning")
    print("=" * 80)

    print(results.to_string(index=False))

    # Find best configuration
    best_row = results.loc[
        results["accuracy"].idxmax()
    ]

    print("\nBest Configuration")
    print("=" * 40)
    print(f"C: {best_row['C']}")
    print(f"Gamma: {best_row['gamma']}")
    print(f"Accuracy: {best_row['accuracy']:.4f}")
    print(f"F1-score: {best_row['f1_score']:.4f}")

    # Save results
    results.to_csv(
        "Day21/Task5/svm_rbf_tuning_results.csv",
        index=False
    )

    # Plot results
    plot_tuning_results(results)


if __name__ == "__main__":
    main()