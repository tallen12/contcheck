"""Interfaces for text analysis and inflection."""

from collections.abc import Iterable, Iterator
from typing import Protocol

from contcheck.analysis.models import AnalyzedText


class TextAnalyzer(Protocol):
    """Splits text into tagged, lemmatized tokens."""

    def analyze(self, text: str) -> AnalyzedText:
        """Analyze one paragraph."""
        ...

    def analyze_many(self, texts: Iterable[str]) -> Iterator[AnalyzedText]:
        """Analyze paragraphs in order; adapters may batch for speed."""
        ...


class Inflector(Protocol):
    """Produces inflected forms of a lemma for a Penn Treebank tag."""

    def inflect(self, lemma: str, tag: str) -> tuple[str, ...]:
        """Return the forms of ``lemma`` for ``tag``, most common first; empty if none."""
        ...


class WordClassLexicon(Protocol):
    """Knows whether a word is listed as an adjective or an adverb."""

    def is_adjective(self, word: str) -> bool:
        """Whether ``word`` is a known adjective, ignoring case."""
        ...

    def is_adverb(self, word: str) -> bool:
        """Whether ``word`` is a known adverb, ignoring case."""
        ...


class Lemmatizer(Protocol):
    """Knows which lemmas a word form can belong to, out of context."""

    def lemmas(self, word: str) -> frozenset[str]:
        """Every lemma ``word`` can have; empty when it is not a known English word."""
        ...
