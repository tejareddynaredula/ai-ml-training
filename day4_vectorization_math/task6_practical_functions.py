import numpy as np


def euclidean_distance_matrix(a, b):
    """Calculate row-wise Euclidean distance without loops."""
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape")

    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def matrix_ops_report(a, b):
    """Return common matrix operations."""
    a = np.asarray(a)
    b = np.asarray(b)

    report = {
        "product": a @ b,
        "transpose": a.T,
    }

    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        report["determinant"] = np.linalg.det(a)

        try:
            report["inverse"] = np.linalg.inv(a)
        except np.linalg.LinAlgError:
            report["inverse"] = None

    return report


def benchmark_loop_vs_vectorized(n):
    """Compare loop and vectorized sum-of-squares performance."""
    import time

    arr = np.arange(n, dtype=float)

    start = time.perf_counter()
    loop_result = sum(x ** 2 for x in arr)
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    vectorized_result = np.sum(arr ** 2)
    vectorized_time = time.perf_counter() - start

    return {
        "loop_time": loop_time,
        "vectorized_time": vectorized_time,
        "speedup": loop_time / vectorized_time,
        "results_match": np.isclose(loop_result, vectorized_result),
    }


def main():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 3], [4, 5]])

    print("Euclidean distances:")
    print(euclidean_distance_matrix(a, b))

    print("\nMatrix operations:")
    report = matrix_ops_report(a, b)
    for key, value in report.items():
        print(f"{key}:\n{value}")

    print("\nBenchmark:")
    print(benchmark_loop_vs_vectorized(1_000_000))


if __name__ == "__main__":
    main()