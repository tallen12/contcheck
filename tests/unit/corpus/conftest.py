"""Fakes used only by corpus tests."""

from collections.abc import Iterator, Sequence

import pytest

from contcheck.corpus.models import Article

PROSE = " ".join(["This sentence is long enough to count as prose for the filter."] * 2)


class FakeArticleSource:
    """An ``ArticleSource`` over a fixed list of articles."""

    def __init__(self, articles: Sequence[Article]) -> None:
        self._articles = articles

    def articles(self) -> Iterator[Article]:
        yield from self._articles


@pytest.fixture
def articles() -> FakeArticleSource:
    return FakeArticleSource(
        [
            Article(id="1", title="Short", text=f"Heading\n\n{PROSE}"),
            Article(id="2", title="Long", text=f"Intro\n\n{PROSE}\n\n- a list item\n\n{PROSE}"),
            Article(id="3", title="More", text=f"{PROSE}\n\n{PROSE}\n\n{PROSE}"),
        ]
    )
