"""Value types for spelling lookups."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Suggestion:
    """A dictionary word close to the looked-up word."""

    term: str
    distance: int
    frequency: int
