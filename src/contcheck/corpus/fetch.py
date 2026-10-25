"""Turning streamed articles into clean raw paragraphs."""

from collections.abc import Iterator
from dataclasses import dataclass

from contcheck.corpus.documents import MIN_PARAGRAPHS, RawParagraph
from contcheck.corpus.models import Article
from contcheck.corpus.protocols import ArticleSource
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


def fetch_paragraphs(
    source: ArticleSource,
    *,
    prefix: str,
    limit: int,
    paragraph_filter: ParagraphFilter,
    per_article: int = 10,
) -> Iterator[RawParagraph]:
    """Up to ``limit`` paragraphs from articles with at least two prose paragraphs.

    Each article becomes one source, ``<prefix>-<article id>``, so documents never mix
    articles. Only the first ``per_article`` paragraphs of an article are used, so long
    articles do not crowd out the rest.
    """
    if limit <= 0:
        return
    count = 0
    for article in source.articles():
        paragraphs = article_paragraphs(article, paragraph_filter)
        if len(paragraphs) < MIN_PARAGRAPHS:
            continue
        for text in paragraphs[:per_article]:
            yield RawParagraph(source_id=f"{prefix}-{article.id}", text=text)
            count += 1
            if count >= limit:
                return
