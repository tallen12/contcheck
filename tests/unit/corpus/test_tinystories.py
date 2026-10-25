from contcheck.corpus.hub import HubQuery
from contcheck.corpus.ids import text_id
from contcheck.corpus.tinystories import QUERY, TinyStoriesArticles

STORY = "Once upon a time there was a cat.\nThe cat liked milk. \n\nThe end."


def test_stories_become_articles_with_one_paragraph_per_line() -> None:
    calls: list[tuple[HubQuery, int]] = []

    def load(query: HubQuery, seed: int) -> list[object]:
        calls.append((query, seed))
        return [{"text": STORY}]

    articles = list(TinyStoriesArticles(load, seed=5).articles())

    assert calls == [(QUERY, 5)]
    assert QUERY.data_files == "TinyStoriesV2-GPT4-train.jsonl"
    assert [a.id for a in articles] == [text_id(STORY)]
    assert (
        articles[0].text == "Once upon a time there was a cat.\n\nThe cat liked milk.\n\nThe end."
    )
