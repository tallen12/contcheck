"""``ArticleSource`` streaming SimpleStories, short stories written by gpt-4o-mini.

The rows come from ``SimpleStories/SimpleStories`` (MIT), one story per row with the
topic, theme, style, persona and grammar feature it was generated with. The stories
vary more in wording than TinyStories and hold much dialogue. Paragraphs are separated
by blank lines, so each story becomes one article as it is.
"""

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict

from contcheck.corpus.hub import HubLoader, HubQuery
from contcheck.corpus.models import Article

QUERY = HubQuery(dataset="SimpleStories/SimpleStories")


class SimpleStoryRow(BaseModel):
    """The fields of a SimpleStories row that the source uses."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    story: str
    generation_id: str
    topic: str


class SimpleStoriesArticles:
    """Streams a shuffled order of SimpleStories stories as articles."""

    def __init__(self, load: HubLoader, *, seed: int, query: HubQuery = QUERY) -> None:
        self._load = load
        self._seed = seed
        self._query = query

    def articles(self) -> Iterator[Article]:
        for raw in self._load(self._query, self._seed):
            row = SimpleStoryRow.model_validate(raw)
            yield Article(id=row.generation_id, title=row.topic, text=row.story)
