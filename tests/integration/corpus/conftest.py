"""Helpers for tests that stream real corpora from the Hugging Face Hub."""

from collections.abc import Iterator
from dataclasses import replace
from itertools import islice

from contcheck.corpus.hub import HubQuery, stream_query
from contcheck.corpus.models import Article
from contcheck.corpus.protocols import ArticleSource


def small_stream(query: HubQuery, seed: int) -> Iterator[object]:
    """Stream ``query`` with a tiny shuffle buffer, so a test reads only a few rows."""
    return stream_query(replace(query, shuffle_buffer=4), seed)


def first_articles(source: ArticleSource, count: int = 2) -> list[Article]:
    return list(islice(source.articles(), count))
