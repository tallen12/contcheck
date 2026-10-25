"""``contcheck corpus``: fetch clean text."""

from enum import StrEnum
from pathlib import Path

from cyclopts import App

from contcheck.corpus.fetch import ParagraphFilter, fetch_paragraphs
from contcheck.corpus.jsonl import write_jsonl
from contcheck.corpus.protocols import ArticleSource
from contcheck.corpus.wikipedia import WikipediaArticles, stream_from_hub


class CorpusSource(StrEnum):
    """Corpora ``corpus fetch`` can stream."""

    WIKIPEDIA = "wikipedia"


def build_corpus_app() -> App:
    """The ``corpus`` command group."""
    app = App(name="corpus", help="Fetch and prepare clean text.")

    @app.command
    def fetch(
        *,
        source: CorpusSource = CorpusSource.WIKIPEDIA,
        limit: int = 60_000,
        out: Path,
        seed: int = 13,
    ) -> None:
        """Stream up to ``limit`` clean prose paragraphs from ``source`` into a JSONL file.

        Parameters
        ----------
        source
            Corpus to stream.
        limit
            Number of paragraphs to write.
        out
            JSONL file of raw paragraphs, one per line.
        seed
            Seed for the article order.
        """
        articles = _article_source(source, seed=seed)
        paragraphs = fetch_paragraphs(
            articles, prefix=source.value, limit=limit, paragraph_filter=ParagraphFilter()
        )
        count = write_jsonl(out, paragraphs)
        print(f"wrote {count} paragraphs to {out}")

    return app


def _article_source(source: CorpusSource, *, seed: int) -> ArticleSource:
    match source:
        case CorpusSource.WIKIPEDIA:
            return WikipediaArticles(stream_from_hub, seed=seed)
