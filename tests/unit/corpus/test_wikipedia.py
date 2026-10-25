from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from contcheck.corpus.wikipedia import DATASET, SNAPSHOT, WikipediaArticles


def test_rows_become_articles_from_the_configured_snapshot() -> None:
    calls: list[tuple[str, str, int]] = []

    def load(dataset: str, snapshot: str, seed: int) -> Iterable[object]:
        calls.append((dataset, snapshot, seed))
        return [{"id": "12", "url": "u", "title": "Anarchism", "text": "A.\n\nB."}]

    articles = list(WikipediaArticles(load, seed=7).articles())

    assert calls == [(DATASET, SNAPSHOT, 7)]
    assert [(a.id, a.title) for a in articles] == [("12", "Anarchism")]


def test_malformed_rows_are_rejected() -> None:
    def load(dataset: str, snapshot: str, seed: int) -> Iterable[object]:
        return [{"id": 12, "title": "x"}]

    with pytest.raises(ValidationError):
        list(WikipediaArticles(load, seed=0).articles())
