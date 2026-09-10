import numpy as np


def statistics_report(arr):
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "variance": np.var(arr),
        "percentile_50": np.percentile(arr, 50),
        "percentile_90": np.percentile(arr, 90),
        "mean_axis_0": np.mean(arr, axis=0),
        "mean_axis_1": np.mean(arr, axis=1),
    }


def main():
    data = np.array([
        [10, 20, 30],
        [20, 30, 40],
        [30, 40, 50]
    ])

    result = statistics_report(data)

    print("Mean:", result["mean"])
    print("Median:", result["median"])
    print("Standard deviation:", result["std"])
    print("Variance:", result["variance"])
    print("50th percentile:", result["percentile_50"])
    print("90th percentile:", result["percentile_90"])
    print("Mean along axis 0:", result["mean_axis_0"])
    print("Mean along axis 1:", result["mean_axis_1"])


if __name__ == "__main__":
    main()