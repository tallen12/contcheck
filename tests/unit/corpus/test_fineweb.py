from contcheck.corpus.fineweb import QUERY, FineWebEduArticles, page_id
from contcheck.corpus.hub import HubQuery


def test_rows_become_articles_with_one_paragraph_per_line() -> None:
    calls: list[tuple[HubQuery, int]] = []

    def load(query: HubQuery, seed: int) -> list[object]:
        calls.append((query, seed))
        return [
            {
                "id": "<urn:uuid:0d8a>",
                "url": "http://example.org/jane",
                "text": "The Independent Jane\nFirst paragraph.\nSecond paragraph.",
                "score": 2.75,
            }
        ]

    articles = list(FineWebEduArticles(load, seed=5).articles())

    assert calls == [(QUERY, 5)]
    assert QUERY.config == "sample-10BT"
    assert [(a.id, a.title) for a in articles] == [("0d8a", "http://example.org/jane")]
    assert articles[0].text == "The Independent Jane\n\nFirst paragraph.\n\nSecond paragraph."


def test_page_id_keeps_plain_ids() -> None:
    assert page_id("abc") == "abc"
