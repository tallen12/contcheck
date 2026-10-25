"""``ArticleSource`` streaming FineWeb-Edu, educational web pages from Common Crawl.

The dataset is ODC-By 1.0 (``HuggingFaceFW/fineweb-edu``). Its text keeps one paragraph
per line, so each line becomes a paragraph of the article.
"""

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict

from contcheck.corpus.cleanup import lines_to_paragraphs
from contcheck.corpus.hub import HubLoader, HubQuery
from contcheck.corpus.models import Article

QUERY = HubQuery(dataset="HuggingFaceFW/fineweb-edu", config="sample-10BT")
UUID_PREFIX = "<urn:uuid:"


class FineWebRow(BaseModel):
    """The fields of a FineWeb-Edu row that the source uses."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    url: str
    text: str


class FineWebEduArticles:
    """Streams a shuffled order of FineWeb-Edu pages as articles."""

    def __init__(self, load: HubLoader, *, seed: int, query: HubQuery = QUERY) -> None:
        self._load = load
        self._seed = seed
        self._query = query

    def articles(self) -> Iterator[Article]:
        for raw in self._load(self._query, self._seed):
            row = FineWebRow.model_validate(raw)
            yield Article(id=page_id(row.id), title=row.url, text=lines_to_paragraphs(row.text))


def page_id(row_id: str) -> str:
    """The bare UUID of a ``<urn:uuid:...>`` row id."""
    return row_id.removeprefix(UUID_PREFIX).removesuffix(">")
