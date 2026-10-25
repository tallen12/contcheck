"""The rule-based generation pipeline: documents in, labeled records out."""

import hashlib
import random
from collections.abc import Iterable, Iterator, Sequence

from contcheck.analysis.models import AnalyzedText, join_words
from contcheck.analysis.protocols import TextAnalyzer
from contcheck.analysis.tense import tense_histogram
from contcheck.corpus.documents import Document
from contcheck.corpus.record import Record, RecordSource, revert_edits
from contcheck.corrupt.apply import corrupt_paragraph
from contcheck.corrupt.config import GenConfig
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.registry import CorruptorRegistry


class RoundTripError(AssertionError):
    """Reverting a record's edits did not give back the clean words: an applier bug."""


def derive_seed(seed: int, doc_id: str, paragraph: int, variant: int) -> int:
    """A seed for one variant of one paragraph, independent of processing order."""
    digest = hashlib.sha256(f"{seed}:{doc_id}:{paragraph}:{variant}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _analyze_documents(
    documents: Sequence[Document], analyzer: TextAnalyzer
) -> Iterator[tuple[Document, list[AnalyzedText]]]:
    # One stream over every paragraph lets the analyzer batch across documents.
    analyzed = analyzer.analyze_many(p for document in documents for p in document.paragraphs)
    for document in documents:
        yield document, [next(analyzed) for _ in document.paragraphs]


def generate_records(
    documents: Iterable[Document],
    *,
    analyzer: TextAnalyzer,
    registry: CorruptorRegistry,
    config: GenConfig,
) -> Iterator[Record]:
    """Yield ``config.variants_per_paragraph`` corrupted records per paragraph.

    The tense profile comes from the whole clean document. Every record is checked to
    round-trip back to its clean words; a failure raises :class:`RoundTripError`.
    """
    for document, paragraphs in _analyze_documents(list(documents), analyzer):
        tense = tense_histogram(token for paragraph in paragraphs for token in paragraph.tokens)
        for index, analyzed in enumerate(paragraphs):
            context = CorruptionContext(analyzed.tokens, tense)
            for variant in range(config.variants_per_paragraph):
                rng = random.Random(derive_seed(config.seed, document.doc_id, index, variant))
                corrupted = corrupt_paragraph(context, registry, config, rng)
                if revert_edits(corrupted.words, corrupted.edits) != context.words:
                    raise RoundTripError(f"{document.doc_id} paragraph {index} variant {variant}")
                yield Record(
                    doc_id=document.doc_id,
                    paragraph=index,
                    variant=variant,
                    split=document.split,
                    source=RecordSource.RULE,
                    clean=analyzed.text,
                    noisy=join_words(corrupted.words, corrupted.spaces),
                    words=corrupted.words,
                    labels=corrupted.labels,
                    edits=corrupted.edits,
                )
