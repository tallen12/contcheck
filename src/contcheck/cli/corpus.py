"""``contcheck corpus``, ``merge`` and ``report``: fetch, combine and inspect data."""

from enum import StrEnum
from itertools import chain
from pathlib import Path
from typing import assert_never

from cyclopts import App

from contcheck.cli.corrupt import RULE_KINDS
from contcheck.corpus.fetch import ParagraphFilter, fetch_paragraphs
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.corpus.merge import merge_records
from contcheck.corpus.protocols import ArticleSource
from contcheck.corpus.record import Record
from contcheck.corpus.report import build_report, format_report
from contcheck.corpus.wikipedia import WikipediaArticles, stream_from_hub
from contcheck.corrupt.config import load_gen_config


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
        case unreachable:
            assert_never(unreachable)


def merge(inputs: list[Path], /, *, out_dir: Path, seed: int = 13) -> None:
    """Merge record files into ``train.jsonl``, ``val.jsonl`` and ``test.jsonl``.

    Repeated records are dropped, each split is shuffled, and every document stays in
    the split it was assigned.

    Parameters
    ----------
    inputs
        Record JSONL files to merge.
    out_dir
        Directory for the per-split files.
    seed
        Seed for the shuffle.
    """
    records = chain.from_iterable(read_jsonl(path, Record) for path in inputs)
    for split, split_records in merge_records(records, seed=seed).items():
        path = out_dir / f"{split.value}.jsonl"
        count = write_jsonl(path, split_records)
        print(f"wrote {count} records to {path}")


def report(path: Path, /, *, config: Path | None = None) -> None:
    """Print per-split, per-kind and errors-per-paragraph counts for a record file.

    Parameters
    ----------
    path
        Record JSONL file.
    config
        Generation config whose targets to compare against.
    """
    summary = build_report(read_jsonl(path, Record))
    if config is None:
        print(format_report(summary))
        return
    gen_config = load_gen_config(config)
    print(
        format_report(
            summary,
            target_errors=gen_config.errors_per_paragraph,
            # Every kind with a corruptor, so one that never fired shows 0 against its target.
            target_kinds={kind: gen_config.weight(kind) for kind in RULE_KINDS},
        )
    )
