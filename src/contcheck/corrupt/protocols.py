"""Interfaces for corruptors and the randomness they draw on."""

from collections.abc import Sequence
from typing import Protocol

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.taxonomy.kinds import ErrorKind


class Rng(Protocol):
    """A seeded source of randomness; ``random.Random`` satisfies it."""

    def random(self) -> float:
        """A float in [0, 1)."""
        ...

    def randrange(self, stop: int, /) -> int:
        """An int in [0, stop)."""
        ...

    def choice[T](self, seq: Sequence[T], /) -> T:
        """One element of a non-empty sequence."""
        ...


class Corruptor(Protocol):
    """Introduces one error of a single kind into a clean paragraph."""

    @property
    def kind(self) -> ErrorKind:
        """The kind of error every edit from this corruptor has."""
        ...

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        """Propose one edit, or ``None`` when the paragraph has no valid site."""
        ...
