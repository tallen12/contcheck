"""Turning streamed articles into clean raw paragraphs."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import assert_never

from contcheck.corpus.cleanliness import ACCEPT_ALL
from contcheck.corpus.documents import MIN_PARAGRAPHS, RawParagraph
from contcheck.corpus.models import Article, Clean, Rejected
from contcheck.corpus.protocols import ArticleSource, ParagraphScreen
from contcheck.corpus.readers import PARAGRAPH_BREAK, normalize_whitespace

SENTENCE_END = (".", "!", "?", '"', "”")


@dataclass(frozen=True, slots=True)
class ParagraphFilter:
    """Keeps prose paragraphs: long enough, not too long, ending like a sentence.

    Headings, list items, tables and captions fail one of these checks.
    """

    min_words: int = 20
    max_words: int = 200

    def accepts(self, paragraph: str) -> bool:
        words = len(paragraph.split())
        return self.min_words <= words <= self.max_words and paragraph.endswith(SENTENCE_END)


def article_paragraphs(article: Article, paragraph_filter: ParagraphFilter) -> list[str]:
    """The prose paragraphs of ``article``, in order, with whitespace normalized."""
    paragraphs = (normalize_whitespace(block) for block in PARAGRAPH_BREAK.split(article.text))
    return [p for p in paragraphs if paragraph_filter.accepts(p)]


def screened_paragraphs(
    article: Article, paragraph_filter: ParagraphFilter, screen: ParagraphScreen, *, limit: int
) -> list[str]:
    """Up to ``limit`` prose paragraphs of ``article`` that ``screen`` accepts, in order.

    Screening stops once ``limit`` paragraphs are kept, so long books cost no more than
    short articles.
    """
    kept: list[str] = []
    for paragraph in article_paragraphs(article, paragraph_filter):
        verdict = screen.screen(paragraph)
        match verdict:
            case Clean():
                kept.append(paragraph)
                if len(kept) >= limit:
                    break
            case Rejected():
                pass
            case _:
                assert_never(verdict)
    return kept


def fetch_paragraphs(
    source: ArticleSource,
    *,
    prefix: str,
    limit: int,
    paragraph_filter: ParagraphFilter,
    screen: ParagraphScreen = ACCEPT_ALL,
    per_article: int = 10,
) -> Iterator[RawParagraph]:
    """Up to ``limit`` paragraphs from articles with at least two clean prose paragraphs.

    Each article becomes one source, ``<prefix>-<article id>``, so documents never mix
    articles. Only the first ``per_article`` paragraphs of an article that pass
    ``paragraph_filter`` and ``screen`` are used, so long articles do not crowd out the
    rest. An article whose id was already used is skipped: sources that hash the text
    into the id give duplicate rows the same id, and a source's paragraphs must stay
    consecutive for document grouping.
    """
    if limit <= 0:
        return
    count = 0
    used: set[str] = set()
    for article in source.articles():
        if article.id in used:
            continue
        used.add(article.id)
        paragraphs = screened_paragraphs(article, paragraph_filter, screen, limit=per_article)
        if len(paragraphs) < MIN_PARAGRAPHS:
            continue
        for text in paragraphs:
            yield RawParagraph(source_id=f"{prefix}-{article.id}", text=text)
            count += 1
            if count >= limit:
                return
