"""``ArticleSource`` streaming the stories subset of Cosmopedia, written by Mixtral.

The rows come from ``HuggingFaceTB/cosmopedia`` (Apache 2.0), config ``stories``: about
five million stories for audiences from young children to adults, much of it in the
first person. The other subsets rewrite web, textbook and how-to text in registers the
other sources already cover, or are mostly lists and markup. Paragraphs are separated by
blank lines, so each story becomes one article as it is.
"""

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict

from contcheck.corpus.hub import HubLoader, HubQuery
from contcheck.corpus.ids import text_id
from contcheck.corpus.models import Article

QUERY = HubQuery(dataset="HuggingFaceTB/cosmopedia", config="stories")


class CosmopediaRow(BaseModel):
    """The fields of a Cosmopedia row that the source uses."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str


class CosmopediaStories:
    """Streams a shuffled order of Cosmopedia stories as articles."""

    def __init__(self, load: HubLoader, *, seed: int, query: HubQuery = QUERY) -> None:
        self._load = load
        self._seed = seed
        self._query = query

    def articles(self) -> Iterator[Article]:
        for raw in self._load(self._query, self._seed):
            row = CosmopediaRow.model_validate(raw)
            yield Article(id=text_id(row.text), title="", text=row.text.strip())
