import numpy as np


def create_identity_matrix(n: int) -> np.ndarray:
    return np.eye(n)


def random_matrix_stats(rows: int, cols: int) -> dict:
    arr = np.random.rand(rows, cols)
    return {
        "min": arr.min(),
        "max": arr.max(),
        "mean": arr.mean(),
        "std": arr.std(),
    }


def reshape_pipeline(arr: np.ndarray, shape: tuple) -> np.ndarray:
    try:
        return arr.reshape(shape)
    except ValueError:
        raise ValueError("Invalid reshape dimensions")


def stack_arrays(arr_list: list[np.ndarray], axis: int) -> np.ndarray:
    return np.stack(arr_list, axis=axis)


def inspect_array(arr: np.ndarray) -> None:
    print(
        f"Shape: {arr.shape} | Dtype: {arr.dtype} | "
        f"Memory: {arr.nbytes} bytes | Ndim: {arr.ndim}"
    )


def main() -> None:
    # Array creation
    print("List:", np.array([1, 2, 3]))
    print("Zeros:\n", np.zeros((2, 3)))
    print("Ones:\n", np.ones((2, 3)))
    print("Arange:", np.arange(0, 10, 2))
    print("Linspace:", np.linspace(0, 1, 5))
    print("Identity:\n", np.eye(3))
    print("Random:\n", np.random.rand(2, 3))

    # Array properties
    print("\nArray Inspector:")
    inspect_array(np.array([1, 2, 3]))
    inspect_array(np.zeros((2, 3)))
    inspect_array(np.ones((2, 2, 2)))

    # Reshape, flatten and ravel
    arr = np.arange(1, 7).reshape(2, 3)
    # reshape changes shape; flatten makes a copy; ravel flattens as a view when possible.
    print("\nReshape:\n", arr)
    print("Flatten:", arr.flatten())
    print("Ravel:", arr.ravel())

    # Concatenation and stacking
    a, b = np.array([1, 2]), np.array([3, 4])
    print("Concatenate:", np.concatenate((a, b)))
    print("Vstack:\n", np.vstack((a, b)))
    print("Hstack:", np.hstack((a, b)))

    # Practical functions
    print("\nIdentity Function:\n", create_identity_matrix(3))
    print("Stats:", random_matrix_stats(2, 3))
    print("Pipeline:\n", reshape_pipeline(np.arange(6), (2, 3)))
    print("Stack Function:\n", stack_arrays([a, b], 0))

    # Dtype and memory comparison
    int_arr = np.ones((1000, 1000), dtype=np.int32)
    float_arr = int_arr.astype(np.float64)
    print("\nint32 memory:", int_arr.nbytes, "bytes")
    print("float64 memory:", float_arr.nbytes, "bytes")


if __name__ == "__main__":
    main()