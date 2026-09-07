from day1_python_core import (
    list_comprehension,
    dict_comprehension,
    flatten_nested_list,
    word_frequency,
    chunk_list,
    fibonacci,
    args_sum,
    kwargs_info,
)


def test_list_comprehension():
    assert list_comprehension([1, 2, 3]) == [1, 4, 9]


def test_empty_list():
    assert list_comprehension([]) == []


def test_single_element():
    assert list_comprehension([5]) == [25]


def test_dict_comprehension():
    assert dict_comprehension([1, 2]) == {1: 1, 2: 4}


def test_flatten():
    assert flatten_nested_list([1, [2, [3]]]) == [1, 2, 3]


def test_word_frequency():
    assert word_frequency("ai ai ml") == {"ai": 2, "ml": 1}


def test_chunk_list():
    assert list(chunk_list([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_fibonacci_zero():
    assert list(fibonacci(0)) == [0]


def test_args_sum():
    assert args_sum(1, 2, 3) == 6


def test_kwargs_info():
    assert kwargs_info(name="Teja") == {"name": "Teja"}