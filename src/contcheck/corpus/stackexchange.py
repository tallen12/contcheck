"""``ArticleSource`` streaming highly voted Stack Exchange answers.

The rows come from ``HuggingFaceH4/stack-exchange-preferences`` (CC BY-SA 4.0): one row
per question, with every answer as HTML and a ``pm_score`` of
``round(log2(1 + upvotes))``, plus 1 when the asker accepted it. Each answer above the
score threshold becomes one article, so a document never mixes authors. Only prose
paragraphs survive the HTML cleanup: code, quotes, lists, headings and tables go.
"""

from collections.abc import Iterable, Iterator, Sequence
from html.parser import HTMLParser
from typing import override
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from contcheck.corpus.cleanup import join_paragraphs
from contcheck.corpus.hub import HubLoader, HubQuery
from contcheck.corpus.models import Article

DATASET = "HuggingFaceH4/stack-exchange-preferences"
SHUFFLE_BUFFER = 1_000
MIN_PM_SCORE = 4
"""About 10 upvotes, or 6 for an accepted answer."""

SITES = (
    "academia",
    "bicycles",
    "cooking",
    "diy",
    "expatriates",
    "fitness",
    "gardening",
    "history",
    "interpersonal",
    "law",
    "lifehacks",
    "literature",
    "money",
    "movies",
    "outdoors",
    "parenting",
    "pets",
    "philosophy",
    "politics",
    "scifi",
    "skeptics",
    "travel",
    "workplace",
    "worldbuilding",
    "writers",
)
"""Sites whose answers are mostly prose. Language-learning sites (``english``, ``ell``)
are left out: their posts quote wrong sentences on purpose."""

SKIPPED_BLOCKS = frozenset(
    {"pre", "blockquote", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "aside"}
)
"""Elements whose text never becomes a paragraph."""

TAINTING_INLINE = frozenset({"code", "kbd", "img", "br", "sup", "sub", "math"})
"""Inline elements that leave a hole or a line break, so their paragraph is dropped."""


def site_query(site: str) -> HubQuery:
    """The files of one ``<site>.stackexchange.com`` site."""
    return HubQuery(
        dataset=DATASET, data_dir=f"data/{site}.stackexchange.com", shuffle_buffer=SHUFFLE_BUFFER
    )


class Answer(BaseModel):
    """The fields of an answer that the source uses."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    answer_id: int
    pm_score: int
    text: str


class QuestionRow(BaseModel):
    """One question with its answers; ``metadata`` starts with the question URL."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    qid: int
    answers: list[Answer]
    metadata: list[str]


class StackExchangeArticles:
    """Streams highly voted answers, taking questions from each site in turn."""

    def __init__(
        self,
        load: HubLoader,
        *,
        seed: int,
        sites: Sequence[str] = SITES,
        min_pm_score: int = MIN_PM_SCORE,
    ) -> None:
        self._load = load
        self._seed = seed
        self._sites = sites
        self._min_pm_score = min_pm_score

    def articles(self) -> Iterator[Article]:
        streams = [self._load(site_query(site), self._seed) for site in self._sites]
        for raw in round_robin(streams):
            row = QuestionRow.model_validate(raw)
            url = row.metadata[0] if row.metadata else ""
            site = urlparse(url).hostname or "unknown"
            for answer in row.answers:
                if answer.pm_score >= self._min_pm_score:
                    yield Article(
                        id=f"{site.split('.')[0]}-{answer.answer_id}",
                        title=url,
                        text=join_paragraphs(html_paragraphs(answer.text)),
                    )


def round_robin(streams: Iterable[Iterable[object]]) -> Iterator[object]:
    """One item from each stream in turn until every stream is exhausted."""
    iterators = [iter(stream) for stream in streams]
    while iterators:
        remaining: list[Iterator[object]] = []
        for iterator in iterators:
            for item in iterator:
                yield item
                remaining.append(iterator)
                break
        iterators = remaining


def html_paragraphs(text: str) -> list[str]:
    """The text of the top-level ``<p>`` elements of an HTML post that are pure prose."""
    parser = _ParagraphParser()
    parser.feed(text)
    parser.close()
    return parser.paragraphs


class _ParagraphParser(HTMLParser):
    """Collects ``<p>`` text outside skipped blocks, dropping paragraphs with holes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._skip_depth = 0
        self._parts: list[str] | None = None
        self._tainted = False

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIPPED_BLOCKS:
            self._skip_depth += 1
        elif tag == "p" and self._skip_depth == 0:
            self._parts = []
            self._tainted = False
        elif tag in TAINTING_INLINE:
            self._tainted = True

    @override
    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in TAINTING_INLINE:
            self._tainted = True

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_BLOCKS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "p" and self._parts is not None:
            paragraph = " ".join("".join(self._parts).split())
            if paragraph and not self._tainted:
                self.paragraphs.append(paragraph)
            self._parts = None

    @override
    def handle_data(self, data: str) -> None:
        if self._parts is not None and self._skip_depth == 0:
            self._parts.append(data)
