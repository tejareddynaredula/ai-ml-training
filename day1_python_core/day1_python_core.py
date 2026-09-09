from collections import Counter
from functools import reduce, wraps
from typing import Any, Callable, Generator
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def list_comprehension(numbers: list[int]) -> list[int]:
    return [x * x for x in numbers]


def dict_comprehension(numbers: list[int]) -> dict[int, int]:
    return {x: x * x for x in numbers}


def set_comprehension(numbers: list[int]) -> set[int]:
    return {x * x for x in numbers}


def lambda_example(x: int) -> int:
    square = lambda n: n * n
    return square(x)


def map_filter_reduce(numbers: list[int]) -> tuple[list[int], list[int], int]:
    mapped = list(map(lambda x: x * 2, numbers))
    filtered = list(filter(lambda x: x % 2 == 0, numbers))
    reduced = reduce(lambda x, y: x + y, numbers, 0)
    return mapped, filtered, reduced


def args_sum(*args: int) -> int:
    return sum(args)


def kwargs_info(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def fibonacci(n: int) -> Generator[int, None, None]:
    a, b = 0, 1

    while a <= n:
        yield a
        a, b = b, a + b


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s executed in %.6f seconds", func.__name__, elapsed)
        return result

    return wrapper


def log_call(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info("Calling %s with args=%s kwargs=%s", func.__name__, args, kwargs)
        return func(*args, **kwargs)

    return wrapper


def flatten_nested_list(data: list[Any]) -> list[Any]:
    return [
        item
        for value in data
        for item in (flatten_nested_list(value) if isinstance(value, list) else [value])
    ]


def word_frequency(text: str) -> dict[str, int]:
    return dict(Counter(text.lower().split()))


def chunk_list(data: list[Any], size: int) -> Generator[list[Any], None, None]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


@timer
@log_call
def decorated_sum(*numbers: int) -> int:
    return sum(numbers)


def main() -> None:
    numbers = [1, 2, 3, 4, 5]

    print("List:", list_comprehension(numbers))
    print("Dictionary:", dict_comprehension(numbers))
    print("Set:", set_comprehension(numbers))
    print("Lambda:", lambda_example(5))

    mapped, filtered, reduced = map_filter_reduce(numbers)
    print("Map:", mapped)
    print("Filter:", filtered)
    print("Reduce:", reduced)

    print("Args Sum:", args_sum(10, 20, 30))
    print("Kwargs:", kwargs_info(name="Teja", role="AI/ML"))
    print("Fibonacci:", list(fibonacci(20)))

    nested = [1, [2, 3], [4, [5, 6]]]
    print("Flattened:", flatten_nested_list(nested))

    text = "python ai python ml ai python"
    print("Word Frequency:", word_frequency(text))

    data = [1, 2, 3, 4, 5, 6, 7]
    print("Chunks:", list(chunk_list(data, 3)))

    print("Decorated Sum:", decorated_sum(10, 20, 30))


if __name__ == "__main__":
    main()
