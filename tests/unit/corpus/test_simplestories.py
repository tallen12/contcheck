from contcheck.corpus.fetch import ParagraphFilter, article_paragraphs
from contcheck.corpus.hub import HubQuery
from contcheck.corpus.simplestories import QUERY, SimpleStoriesArticles

STORY = "Eagerly, Kim went hiking. \n\nThey found a passage.\n\n"


def test_rows_become_articles_keyed_by_generation_id() -> None:
    calls: list[tuple[HubQuery, int]] = []

    def load(query: HubQuery, seed: int) -> list[object]:
        calls.append((query, seed))
        return [{"story": STORY, "generation_id": "7482-6", "topic": "caves", "word_count": 7}]

    articles = list(SimpleStoriesArticles(load, seed=5).articles())

    assert calls == [(QUERY, 5)]
    assert [(a.id, a.title) for a in articles] == [("7482-6", "caves")]
    assert article_paragraphs(articles[0], ParagraphFilter(min_words=1)) == [
        "Eagerly, Kim went hiking.",
        "They found a passage.",
    ]
