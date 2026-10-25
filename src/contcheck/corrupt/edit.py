"""The edit a corruptor proposes on clean text."""

from dataclasses import dataclass
from typing import assert_never

from contcheck.taxonomy.kinds import ErrorKind, Formality, Operation


@dataclass(frozen=True, slots=True)
class Edit:
    """Replace clean words ``start`` to ``end`` (exclusive), which are ``orig``, by ``new``.

    Indices are in the clean words. ``op`` describes the noisy result the way ERRANT
    would: deleting words makes them Missing, inserting words makes them Unnecessary.
    ``formality`` is ``INFORMAL`` when the noisy text is only nonstandard in formal writing.
    """

    start: int
    end: int
    kind: ErrorKind
    op: Operation
    orig: tuple[str, ...]
    new: tuple[str, ...]
    formality: Formality = Formality.ERROR

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid edit range [{self.start}, {self.end})")
        if self.end - self.start != len(self.orig):
            raise ValueError("orig must hold exactly the words in the edit range")
        match self.op:
            case Operation.MISSING:
                valid = bool(self.orig) and not self.new
            case Operation.UNNECESSARY:
                valid = not self.orig and bool(self.new)
            case Operation.REPLACEMENT:
                valid = bool(self.orig) and bool(self.new) and self.orig != self.new
            case unreachable:
                assert_never(unreachable)
        if not valid:
            raise ValueError(f"orig {self.orig} and new {self.new} do not fit op {self.op}")
