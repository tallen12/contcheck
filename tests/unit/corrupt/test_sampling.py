import random

from contcheck.corrupt.sampling import shuffled


def test_shuffled_keeps_every_item_and_leaves_the_input_alone() -> None:
    items = [1, 2, 3, 4, 5]

    result = shuffled(items, random.Random(3))

    assert sorted(result) == items
    assert items == [1, 2, 3, 4, 5]


def test_shuffled_reaches_every_order_of_three() -> None:
    orders = {tuple(shuffled("abc", random.Random(seed))) for seed in range(200)}

    assert len(orders) == 6


def test_shuffled_handles_empty_and_single_inputs() -> None:
    assert shuffled([], random.Random(0)) == []
    assert shuffled(["x"], random.Random(0)) == ["x"]
