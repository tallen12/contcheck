"""``ArticleSource`` streaming 20th-century English fiction from Project Gutenberg.

The rows come from ``sedthh/gutenberg_english``: books scraped from Project Gutenberg's
HTML editions with the license header removed, plus catalogue metadata. The books are in
the public domain in the United States. Each kept book becomes one article.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, Json

from contcheck.corpus.cleanup import join_paragraphs
from contcheck.corpus.hub import HubLoader, HubQuery
from contcheck.corpus.models import Article

QUERY = HubQuery(dataset="sedthh/gutenberg_english", shuffle_buffer=64)
"""Rows are whole books, so the shuffle buffer stays small."""

MIN_AUTHOR_BIRTH_YEAR = 1870
"""Authors born from 1870 on published mostly after 1900, in modern English."""

START_SHARE = 0.1
"""Share of a book's paragraphs skipped as front matter: contents, preface, title pages."""

LINE_BREAKS = re.compile(r"\n+")
START_MARKER = re.compile(r"^\*{3}\s*START OF.*$", re.IGNORECASE | re.MULTILINE)
END_MARKER = re.compile(
    r"^(?:\*{3}\s*END OF|End of (?:the )?Project Gutenberg).*$", re.IGNORECASE | re.MULTILINE
)
NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    "|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last"
)
HEADING = re.compile(
    r"^(?:(?:chapter|book|part|volume|canto|section|act|scene)\s+"
    rf"(?:[IVXLC]+|\d+|(?:the\s+)?(?:{NUMBER_WORDS}))\b|[IVXLC]+\.?$|\d+\.?$)",
    re.IGNORECASE,
)
BRACKETED_NOTE = re.compile(
    r"\[(?:\d+|[ivxlc]+|[A-Z]|Pg\.? ?\d+|Page \d+|Illustration[^\]]*|Footnote[^\]]*)\]",
    re.IGNORECASE,
)
EMPHASIS = re.compile(r"(?<!\w)[_=](\S(?:.*?\S)?)[_=](?!\w)")
AUTHOR_YEARS = re.compile(r",\s*(?:(\d{3,4})\??)?\s*-\s*(?:\d{3,4}\??)?\s*$")
EXCLUDED_FORMS = ("drama", "poetry", "plays")


class GutenbergMetadata(BaseModel):
    """Catalogue fields of one book; ``authors`` holds ``Name, birth-death`` items."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    text_id: int
    title: str
    language: str = "en"
    authors: str = ""
    subjects: str = ""
    bookshelves: str = ""


class GutenbergRow(BaseModel):
    """One row of the dataset: the book text and its metadata as a JSON string."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str = Field(alias="TEXT")
    metadata: Json[GutenbergMetadata] = Field(alias="METADATA")


@dataclass(frozen=True, slots=True)
class BookSelection:
    """Which books count as modern fiction.

    A book is kept when every main author (not an editor, translator or illustrator) has
    a known birth year of at least ``min_author_birth_year``, and its subjects or
    bookshelves name fiction but not drama or poetry. The catalogue has no publication
    date; author birth years are the closest field to it.
    """

    min_author_birth_year: int = MIN_AUTHOR_BIRTH_YEAR

    def accepts(self, metadata: GutenbergMetadata) -> bool:
        if metadata.language != "en":
            return False
        births = author_birth_years(metadata.authors)
        if not births or any(year is None or year < self.min_author_birth_year for year in births):
            return False
        genres = f"{metadata.subjects} {metadata.bookshelves}".lower()
        return "fiction" in genres and not any(form in genres for form in EXCLUDED_FORMS)


def author_birth_years(authors: str) -> list[int | None]:
    """Birth years of the main authors in a catalogue ``authors`` field, in order.

    Contributors with a bracketed role such as ``[Illustrator]`` are skipped. An author
    without a parsable birth year gives ``None``.
    """
    years: list[int | None] = []
    for author in authors.split(";"):
        author = author.strip()
        if not author or "[" in author:
            continue
        match = AUTHOR_YEARS.search(author)
        years.append(int(match.group(1)) if match and match.group(1) else None)
    return years


class GutenbergArticles:
    """Streams a shuffled order of modern English fiction books as articles."""

    def __init__(
        self,
        load: HubLoader,
        *,
        seed: int,
        selection: BookSelection,
        query: HubQuery = QUERY,
        start_share: float = START_SHARE,
    ) -> None:
        self._load = load
        self._seed = seed
        self._selection = selection
        self._query = query
        self._start_share = start_share

    def articles(self) -> Iterator[Article]:
        for raw in self._load(self._query, self._seed):
            row = GutenbergRow.model_validate(raw)
            if not self._selection.accepts(row.metadata):
                continue
            paragraphs = book_paragraphs(row.text)
            start = int(len(paragraphs) * self._start_share)
            yield Article(
                id=str(row.metadata.text_id),
                title=row.metadata.title,
                text=join_paragraphs(paragraphs[start:]),
            )


def book_paragraphs(text: str) -> list[str]:
    """The body paragraphs of a book, unwrapped, without headings or editorial notes.

    In this dataset a single line break ends a paragraph, a doubled one is a hard wrap
    inside a paragraph, and three or more separate sections.
    """
    body = strip_boilerplate(text.replace("\r\n", "\n").replace("\r", "\n"))
    unwrapped = LINE_BREAKS.sub(lambda m: " " if len(m.group()) == 2 else "\n", body)
    return [clean for line in unwrapped.split("\n") if (clean := clean_paragraph(line))]


def strip_boilerplate(text: str) -> str:
    """Cut a Project Gutenberg license header and footer when the text still has them."""
    if start := START_MARKER.search(text):
        text = text[start.end() :]
    if end := END_MARKER.search(text):
        text = text[: end.start()]
    return text


def clean_paragraph(paragraph: str) -> str:
    """Drop a heading, page and footnote markers and ``_emphasis_`` underscores.

    Returns an empty string for a paragraph that is only a heading.
    """
    paragraph = " ".join(paragraph.split())
    if HEADING.match(paragraph) and len(paragraph.split()) <= 8:
        return ""
    paragraph = BRACKETED_NOTE.sub("", paragraph)
    paragraph = EMPHASIS.sub(r"\1", paragraph)
    return " ".join(paragraph.split())
