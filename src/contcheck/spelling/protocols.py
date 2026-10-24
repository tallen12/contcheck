"""Interfaces for dictionary lookups."""

from typing import Protocol

from contcheck.spelling.models import Suggestion


class Dictionary(Protocol):
    """Knows which words are spelled correctly and what a misspelling might mean."""

    def contains(self, word: str) -> bool:
        """Whether ``word`` is a known word, ignoring case."""
        ...

    def suggestions(self, word: str, *, max_distance: int = 2) -> list[Suggestion]:
        """Known words within ``max_distance`` edits, closest and most frequent first."""
        ...
