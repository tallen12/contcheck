"""``contcheck corpus``, ``merge`` and ``report``: fetch, annotate, combine and inspect data."""

import os
import time
from enum import StrEnum
from itertools import chain
from pathlib import Path
from typing import Annotated, assert_never

import httpx
from cyclopts import App, Parameter

from contcheck.cli.corrupt import RULE_KINDS
from contcheck.corpus.annotate import ParagraphAnnotator, RunSettings, annotate_files
from contcheck.corpus.cleanliness import CleanlinessConfig, CleanlinessFilter, CountingScreen
from contcheck.corpus.cosmopedia import CosmopediaStories
from contcheck.corpus.fetch import ParagraphFilter, fetch_paragraphs
from contcheck.corpus.fineweb import FineWebEduArticles
from contcheck.corpus.gutenberg import BookSelection, GutenbergArticles
from contcheck.corpus.hub import stream_query
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.corpus.merge import merge_records
from contcheck.corpus.protocols import ArticleSource
from contcheck.corpus.record import Record
from contcheck.corpus.report import build_report, format_report
from contcheck.corpus.simplestories import SimpleStoriesArticles
from contcheck.corpus.stackexchange import StackExchangeArticles
from contcheck.corpus.tinystories import TinyStoriesArticles
from contcheck.corpus.wikimedia import Wiki, WikimediaArticles
from contcheck.corpus.wikinews import WikinewsArticles
from contcheck.corpus.wikipedia import SIMPLE_SNAPSHOT, WikipediaArticles, stream_from_hub
from contcheck.corrupt.config import load_gen_config
from contcheck.llm.models import ClientConfig
from contcheck.llm.openai_chat import OpenAIChatClient
from contcheck.spelling.symspell_dictionary import load_symspell_dictionary

DEFAULT_LLM_MODEL = "qwen3-coder:30b"
BASE_URL_ENV = "CONTCHECK_LLM_BASE_URL"
MODEL_ENV = "CONTCHECK_LLM_MODEL"
API_KEY_ENV = "CONTCHECK_LLM_API_KEY"


class CorpusSource(StrEnum):
    """Corpora ``corpus fetch`` can stream."""

    WIKIPEDIA = "wikipedia"
    SIMPLE_WIKIPEDIA = "simple-wikipedia"
    GUTENBERG = "gutenberg"
    FINEWEB_EDU = "fineweb-edu"
    WIKINEWS = "wikinews"
    WIKIBOOKS = "wikibooks"
    WIKIVOYAGE = "wikivoyage"
    STACKEXCHANGE = "stackexchange"
    TINYSTORIES = "tinystories"
    SIMPLESTORIES = "simplestories"
    COSMOPEDIA_STORIES = "cosmopedia-stories"


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
        per_article: int = 10,
        min_words: int = 20,
        max_non_word_share: float = 0.03,
        max_sentence_words: int = 70,
    ) -> None:
        """Stream up to ``limit`` clean prose paragraphs from ``source`` into a JSONL file.

        Every paragraph passes the cleanliness checks (dictionary coverage, markup, lists,
        capitals, sentence length); how many each check dropped is printed at the end.

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
        per_article
            Most paragraphs taken from one article, book or answer.
        min_words
            Fewest words in a kept paragraph.
        max_non_word_share
            Largest share of lowercase words missing from the dictionary.
        max_sentence_words
            Most words in any sentence of a kept paragraph.
        """
        articles = article_source(source, seed=seed)
        cleanliness = CleanlinessConfig(
            max_non_word_share=max_non_word_share, max_sentence_words=max_sentence_words
        )
        screen = CountingScreen(CleanlinessFilter(load_symspell_dictionary(), cleanliness))
        paragraphs = fetch_paragraphs(
            articles,
            prefix=source.value,
            limit=limit,
            paragraph_filter=ParagraphFilter(min_words=min_words),
            screen=screen,
            per_article=per_article,
        )
        count = write_jsonl(out, paragraphs)
        print(f"wrote {count} paragraphs to {out}")
        print(f"cleanliness: {screen.summary()}")

    @app.command
    def annotate(
        *,
        source: Annotated[Path, Parameter(name="--in")],
        out: Path,
        model: Annotated[str, Parameter(env_var=MODEL_ENV)] = DEFAULT_LLM_MODEL,
        base_url: Annotated[str | None, Parameter(env_var=BASE_URL_ENV)] = None,
        api_key_env: str = API_KEY_ENV,
        concurrency: int = 4,
        limit: int | None = None,
        timeout: float = 120.0,
        retries: int = 3,
        attempts: int = 3,
        reasoning_switch: bool = False,
    ) -> None:
        """Ask a local LLM which changes it would make to each clean paragraph.

        Writes one JSONL line per paragraph to ``out/<input name>`` with the changes
        (each typed error or informal) and the corrected text; failures go to
        ``out/failed/``. Paragraphs already in the output are skipped, so rerunning
        resumes.

        Parameters
        ----------
        source
            Raw paragraph JSONL file, or a folder of them.
        out
            Folder for the annotated files.
        model
            Model name on the server.
        base_url
            OpenAI-compatible API root, such as ``http://localhost:11434/v1``.
        api_key_env
            Environment variable holding an API key, sent as a bearer token when set.
        concurrency
            Requests in flight at once; match the server's parallel slots.
        limit
            Most new paragraphs to annotate, across all files.
        timeout
            Seconds to wait for one reply.
        retries
            Retries after a connection error, timeout, or a 429, 500, 502, 503 or 504 reply.
        attempts
            Tries per paragraph when the reply is not valid JSON or quotes text the
            paragraph lacks.
        reasoning_switch
            Send ``reasoning_effort: "none"`` to turn off thinking (Ollama with qwen3).
            Leave off for servers that reject the field.
        """
        if base_url is None:
            raise SystemExit(f"pass --base-url or set {BASE_URL_ENV}")
        key = os.environ.get(api_key_env, "")
        api_key = key if key else None
        config = ClientConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            retries=retries,
            concurrency=concurrency,
            reasoning_switch=reasoning_switch,
        )
        auth = "no API key" if api_key is None else f"bearer token from ${api_key_env}"
        print(f"annotating with {model} at {base_url} ({auth}), {concurrency} in parallel")
        with httpx.Client() as http:
            annotator = ParagraphAnnotator(
                OpenAIChatClient(http, config), model=model, attempts=attempts
            )
            annotate_files(
                source,
                out,
                annotator.annotate,
                settings=RunSettings(concurrency=concurrency, limit=limit),
                clock=time.monotonic,
                log=print,
            )

    return app


def article_source(source: CorpusSource, *, seed: int) -> ArticleSource:
    """The ``ArticleSource`` that streams ``source`` from the Hugging Face Hub."""
    match source:
        case CorpusSource.WIKIPEDIA:
            return WikipediaArticles(stream_from_hub, seed=seed)
        case CorpusSource.SIMPLE_WIKIPEDIA:
            return WikipediaArticles(stream_from_hub, seed=seed, snapshot=SIMPLE_SNAPSHOT)
        case CorpusSource.GUTENBERG:
            return GutenbergArticles(stream_query, seed=seed, selection=BookSelection())
        case CorpusSource.FINEWEB_EDU:
            return FineWebEduArticles(stream_query, seed=seed)
        case CorpusSource.WIKINEWS:
            return WikinewsArticles(stream_query, seed=seed)
        case CorpusSource.WIKIBOOKS:
            return WikimediaArticles(stream_query, seed=seed, wiki=Wiki.WIKIBOOKS)
        case CorpusSource.WIKIVOYAGE:
            return WikimediaArticles(stream_query, seed=seed, wiki=Wiki.WIKIVOYAGE)
        case CorpusSource.STACKEXCHANGE:
            return StackExchangeArticles(stream_query, seed=seed)
        case CorpusSource.TINYSTORIES:
            return TinyStoriesArticles(stream_query, seed=seed)
        case CorpusSource.SIMPLESTORIES:
            return SimpleStoriesArticles(stream_query, seed=seed)
        case CorpusSource.COSMOPEDIA_STORIES:
            return CosmopediaStories(stream_query, seed=seed)
        case _:
            assert_never(source)


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
