"""Word labels onto subwords, and subwords into overlapping windows."""

from collections.abc import Sequence

from contcheck.corpus.record import Record
from contcheck.model.models import IGNORE_INDEX, Encoding, Window
from contcheck.model.protocols import Tokenizer
from contcheck.taxonomy.labels import LABEL2ID

SPECIAL_TOKENS = 2
"""Each window starts with the CLS token and ends with the SEP token."""


def align_labels(encoding: Encoding, word_labels: Sequence[int]) -> tuple[int, ...]:
    """Put each word's label on its first subword and :data:`IGNORE_INDEX` elsewhere."""
    labels: list[int] = []
    previous: int | None = None
    for word_id in encoding.word_ids:
        labels.append(word_labels[word_id] if word_id != previous else IGNORE_INDEX)
        previous = word_id
    return tuple(labels)


def window_starts(length: int, *, size: int, stride: int) -> list[int]:
    """Start offsets of windows of ``size`` tokens, ``stride`` apart, covering ``length``.

    The last window is moved back to end exactly at ``length``, so no window is mostly
    padding and every token is covered.
    """
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts


def make_windows(
    encoding: Encoding,
    token_labels: Sequence[int],
    tokenizer: Tokenizer,
    *,
    doc_id: str,
    window: int,
    stride: int,
) -> list[Window]:
    """Cut a paragraph's subwords into windows of at most ``window`` tokens.

    A window that starts inside a word holds that word's later subwords with
    :data:`IGNORE_INDEX`; the word is labeled in the window holding its first subword.
    """
    size = window - SPECIAL_TOKENS
    windows: list[Window] = []
    for start in window_starts(len(encoding.input_ids), size=size, stride=stride):
        end = start + size
        windows.append(
            Window(
                doc_id=doc_id,
                input_ids=(tokenizer.cls_id, *encoding.input_ids[start:end], tokenizer.sep_id),
                word_ids=(None, *encoding.word_ids[start:end], None),
                labels=(IGNORE_INDEX, *token_labels[start:end], IGNORE_INDEX),
                start=start,
            )
        )
    return windows


def record_windows(
    record: Record, tokenizer: Tokenizer, *, window: int, stride: int
) -> list[Window]:
    """Training windows for one record's noisy words and labels."""
    encoding = tokenizer.encode_words(record.words)
    labels = align_labels(encoding, [LABEL2ID[label] for label in record.labels])
    return make_windows(
        encoding, labels, tokenizer, doc_id=record.doc_id, window=window, stride=stride
    )
