"""Seeded sampling helpers built only on :class:`Rng`, so any seeded source works."""

from collections.abc import Iterable

from contcheck.corrupt.protocols import Rng


def shuffled[T](items: Iterable[T], rng: Rng) -> list[T]:
    """A new list of ``items`` in random order (Fisher-Yates)."""
    result = list(items)
    for i in range(len(result) - 1, 0, -1):
        j = rng.randrange(i + 1)
        result[i], result[j] = result[j], result[i]
    return result
