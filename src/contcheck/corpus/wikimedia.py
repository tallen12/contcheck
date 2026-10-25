"""``ArticleSource`` streaming Wikibooks or Wikivoyage pages from the Common Pile.

The rows come from ``common-pile/wikimedia`` (CC BY-SA), plain text converted from
English wikitext dumps, one file set per wiki. Only main-namespace pages are used.
The conversion drops links and templates it cannot render and leaves gaps such as
``the  of Brazil,,``; lines with such gaps are dropped rather than repaired.
"""

import html
from collections.abc import Iterator
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from contcheck.corpus.cleanup import has_dropped_text, join_paragraphs
from contcheck.corpus.hub import HubLoader, HubQuery
from contcheck.corpus.models import Article

DATASET = "common-pile/wikimedia"
MAIN_NAMESPACE = "0"


class Wiki(StrEnum):
    """Wikis of the Common Pile Wikimedia dump that this source reads."""

    WIKIBOOKS = "wikibooks"
    WIKIVOYAGE = "wikivoyage"


def wiki_query(wiki: Wiki) -> HubQuery:
    """The files of one wiki in the dump."""
    return HubQuery(dataset=DATASET, data_files=f"*_{wiki.value}.com.jsonl.gz")


class WikiPageMetadata(BaseModel):
    """The metadata fields of a page that the source uses."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    namespace: str
    title: str


class WikiPageRow(BaseModel):
    """One page of the dump."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    text: str
    metadata: WikiPageMetadata


class WikimediaArticles:
    """Streams a shuffled order of main-namespace pages of one wiki."""

    def __init__(self, load: HubLoader, *, seed: int, wiki: Wiki) -> None:
        self._load = load
        self._seed = seed
        self._query = wiki_query(wiki)

    def articles(self) -> Iterator[Article]:
        for raw in self._load(self._query, self._seed):
            row = WikiPageRow.model_validate(raw)
            if row.metadata.namespace == MAIN_NAMESPACE:
                title = row.metadata.title
                yield Article(id=row.id, title=title, text=page_text(row.text, title))


def page_text(text: str, title: str) -> str:
    """Paragraphs of a page: one per line, without the title line or lines with gaps.

    Every paragraph of the dump is a single line, and section headings sit on the line
    directly above their first paragraph.
    """
    lines = [html.unescape(line).strip() for line in text.splitlines()]
    if lines and lines[0] == title:
        lines = lines[1:]
    return join_paragraphs(line for line in lines if line and not has_dropped_text(line))
