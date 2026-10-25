"""Interfaces for sources of clean text."""

from collections.abc import Iterator
from typing import Protocol

from contcheck.corpus.models import Article, Verdict


class ArticleSource(Protocol):
    """Streams whole articles from a corpus."""

    def articles(self) -> Iterator[Article]:
        """Yield articles in a fixed order for a given source configuration."""
        ...


class ParagraphScreen(Protocol):
    """Decides whether a paragraph is clean enough to train on."""

    def screen(self, paragraph: str) -> Verdict:
        """``Clean`` to keep ``paragraph``, or ``Rejected`` with the failed check."""
        ...
