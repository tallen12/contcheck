import pytest
from pydantic import ValidationError

from contcheck.corpus.hub import HubQuery
from contcheck.corpus.wikinews import QUERY, WikinewsArticles


def row(pageid: int, lang: str) -> dict[str, object]:
    return {
        "pageid": pageid,
        "lang": lang,
        "title": f"Story {pageid}",
        "text": ["First paragraph.", "Second paragraph."],
        "type": "title",
    }


def test_only_english_rows_become_articles() -> None:
    def load(query: HubQuery, seed: int) -> list[object]:
        assert (query, seed) == (QUERY, 2)
        return [row(1, "en"), row(1, "ta"), row(2, "en")]

    articles = list(WikinewsArticles(load, seed=2).articles())

    assert [a.id for a in articles] == ["1", "2"]
    assert articles[0].text == "First paragraph.\n\nSecond paragraph."


def test_malformed_rows_are_rejected() -> None:
    def load(query: HubQuery, seed: int) -> list[object]:
        return [{"pageid": 1, "lang": "en", "title": "t", "text": "not a list"}]

    with pytest.raises(ValidationError):
        list(WikinewsArticles(load, seed=0).articles())
