"""Interface for looking up dictionary words by their Penn Treebank tag."""

from typing import Protocol


class Lexicon(Protocol):
    """Knows which dictionary words are which inflection of which lemma."""

    def lemmas(self, word: str, tag: str) -> frozenset[str]:
        """Lemmas of which lowercase ``word`` is the ``tag`` form, such as ``{"run"}`` for
        ``ran`` and VBD. Empty when ``word`` is not a known word with that tag."""
        ...
