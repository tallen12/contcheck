"""Interfaces for sources of clean text."""

from collections.abc import Iterator
from typing import Protocol

from contcheck.corpus.models import Article


class ArticleSource(Protocol):
    """Streams whole articles from a corpus."""

    def articles(self) -> Iterator[Article]:
        """Yield articles in a fixed order for a given source configuration."""
        ...
