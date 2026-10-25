from contcheck.corpus.cosmopedia import QUERY, CosmopediaStories
from contcheck.corpus.fetch import ParagraphFilter, article_paragraphs
from contcheck.corpus.hub import HubQuery
from contcheck.corpus.ids import text_id

STORY = " In a small town lived Ana.\n\nShe baked bread every day."


def test_stories_become_articles_with_hashed_ids() -> None:
    calls: list[tuple[HubQuery, int]] = []

    def load(query: HubQuery, seed: int) -> list[object]:
        calls.append((query, seed))
        return [{"text": STORY, "format": "story_children", "audience": "young_children"}]

    articles = list(CosmopediaStories(load, seed=5).articles())

    assert calls == [(QUERY, 5)]
    assert QUERY.config == "stories"
    assert [(a.id, a.title) for a in articles] == [(text_id(STORY), "story_children")]
    assert article_paragraphs(articles[0], ParagraphFilter(min_words=1)) == [
        "In a small town lived Ana.",
        "She baked bread every day.",
    ]
