"""Conversions between error spans and word-level BIO labels.

Spans index words with an exclusive end. Every span covers at least one word, including
Missing errors, which have no word of their own: they are anchored on a neighbouring word
with :func:`missing_anchor`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

from contcheck.taxonomy.kinds import ErrorKind
from contcheck.taxonomy.labels import (
    OUTSIDE,
    Outside,
    Prefix,
    Tagged,
    format_label,
    parse_label,
)


@dataclass(frozen=True, slots=True)
class WordSpan:
    """Words ``start`` up to, not including, ``end`` carry an error of ``kind``."""

    start: int
    end: int
    kind: ErrorKind

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")


def spans_to_labels(spans: Sequence[WordSpan], *, word_count: int) -> list[str]:
    """Label ``word_count`` words with ``spans``.

    Raises ``ValueError`` if ``word_count`` is negative, a span runs past the last word or
    two spans overlap, since one word carries one label.
    """
    if word_count < 0:
        raise ValueError(f"negative word count {word_count}")
    labels = [OUTSIDE] * word_count
    for span in spans:
        if span.end > word_count:
            raise ValueError(f"span [{span.start}, {span.end}) exceeds {word_count} words")
        if any(label != OUTSIDE for label in labels[span.start : span.end]):
            raise ValueError(f"span [{span.start}, {span.end}) overlaps another span")
        labels[span.start] = format_label(Tagged(Prefix.BEGIN, span.kind))
        for i in range(span.start + 1, span.end):
            labels[i] = format_label(Tagged(Prefix.INSIDE, span.kind))
    return labels


def labels_to_spans(labels: Sequence[str]) -> list[WordSpan]:
    """Group BIO labels into spans, repairing invalid sequences.

    An ``I-X`` that does not continue a span of kind ``X`` starts a new span, as if it
    were ``B-X``. Raises ``ValueError`` for unknown labels.
    """
    spans: list[WordSpan] = []
    start: int | None = None
    kind: ErrorKind | None = None

    def close(end: int) -> None:
        if start is not None and kind is not None:
            spans.append(WordSpan(start, end, kind))

    for i, text in enumerate(labels):
        match parse_label(text):
            case Outside():
                close(i)
                start, kind = None, None
            case Tagged(prefix=Prefix.INSIDE, kind=current) if current == kind:
                pass
            case Tagged(kind=current):
                close(i)
                start, kind = i, current
            case unreachable:
                assert_never(unreachable)
    close(len(labels))
    return spans


def missing_anchor(gap: int, *, word_count: int) -> int:
    """Return the word that carries a Missing error at ``gap``.

    ``gap`` is the index the missing word would have in the noisy words. The error tags
    the word after the gap, or the word before it when the gap is at the end.

    ``word_count`` words have ``word_count + 1`` gaps, so two gaps always share a word:
    the last two, ``word_count - 1`` and ``word_count``. Errors at both, or a Missing
    error next to another error on its anchor, overlap in :func:`spans_to_labels`.
    Callers must not place such edits together.
    """
    if word_count <= 0:
        raise ValueError("a Missing error needs at least one word to anchor on")
    if not 0 <= gap <= word_count:
        raise ValueError(f"gap {gap} outside 0..{word_count}")
    return gap if gap < word_count else word_count - 1
