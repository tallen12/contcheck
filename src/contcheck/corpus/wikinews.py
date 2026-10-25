"""``ArticleSource`` streaming English Wikinews articles.

The rows come from ``Fumika/Wikinews-multilingual`` (CC BY 2.5, the Wikinews license),
which stores each article as a list of clean paragraphs with links kept as text. Only
English articles are used.
"""

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict

from contcheck.corpus.cleanup import join_paragraphs
from contcheck.corpus.hub import HubLoader, HubQuery
from contcheck.corpus.models import Article

QUERY = HubQuery(dataset="Fumika/Wikinews-multilingual")
LANGUAGE = "en"


class WikinewsRow(BaseModel):
    """The fields of a Wikinews row that the source uses."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    pageid: int
    lang: str
    title: str
    text: list[str]


class WikinewsArticles:
    """Streams a shuffled order of English Wikinews articles."""

    def __init__(self, load: HubLoader, *, seed: int, query: HubQuery = QUERY) -> None:
        self._load = load
        self._seed = seed
        self._query = query

    def articles(self) -> Iterator[Article]:
        for raw in self._load(self._query, self._seed):
            row = WikinewsRow.model_validate(raw)
            if row.lang == LANGUAGE:
                yield Article(id=str(row.pageid), title=row.title, text=join_paragraphs(row.text))
