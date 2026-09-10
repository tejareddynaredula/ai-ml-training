import numpy as np
import time


def benchmark_loop_vs_vectorized(n):
    arr = np.arange(n, dtype=float)

    start = time.perf_counter()
    loop_result = sum(x ** 2 for x in arr)
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    vectorized_result = np.sum(arr ** 2)
    vectorized_time = time.perf_counter() - start

    speedup = loop_time / vectorized_time

    return {
        "loop_time": loop_time,
        "vectorized_time": vectorized_time,
        "speedup": speedup,
        "results_match": np.isclose(loop_result, vectorized_result),
    }


def main():
    result = benchmark_loop_vs_vectorized(1_000_000)

    print("Benchmark: 1,000,000 elements")
    print(f"Loop time:        {result['loop_time']:.6f} seconds")
    print(f"Vectorized time:  {result['vectorized_time']:.6f} seconds")
    print(f"Speedup:          {result['speedup']:.2f}x")
    print(f"Results match:    {result['results_match']}")


if __name__ == "__main__":
    main()