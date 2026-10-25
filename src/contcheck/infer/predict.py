"""Running a ``Tagger`` over many paragraphs of pre-split words."""

from collections.abc import Sequence

import numpy as np

from contcheck.infer.decode import FloatArray, word_probabilities
from contcheck.model.align import align_labels, make_windows
from contcheck.model.models import Window
from contcheck.model.protocols import Tagger, Tokenizer
from contcheck.taxonomy.labels import LABEL2ID, OUTSIDE


def inference_windows(
    words: Sequence[str], tokenizer: Tokenizer, *, window: int, stride: int
) -> list[Window]:
    """Windows for one paragraph, with placeholder labels marking first subwords."""
    encoding = tokenizer.encode_words(words)
    placeholder = align_labels(encoding, [LABEL2ID[OUTSIDE]] * len(words))
    return make_windows(encoding, placeholder, tokenizer, doc_id="", window=window, stride=stride)


def predict_paragraphs(
    paragraphs: Sequence[Sequence[str]],
    *,
    tagger: Tagger,
    tokenizer: Tokenizer,
    window: int,
    stride: int,
    batch_size: int,
) -> list[FloatArray]:
    """Label probabilities ``(words, labels)`` for each paragraph.

    Windows of all paragraphs are batched together, sorted by length so batches carry
    little padding.
    """
    windows = [
        inference_windows(words, tokenizer, window=window, stride=stride) for words in paragraphs
    ]
    flat = [(p, w) for p, paragraph_windows in enumerate(windows) for w in paragraph_windows]
    order = sorted(range(len(flat)), key=lambda i: len(flat[i][1].input_ids))
    logits: list[FloatArray | None] = [None] * len(flat)
    for batch_start in range(0, len(order), batch_size):
        indices = order[batch_start : batch_start + batch_size]
        length = max(len(flat[i][1].input_ids) for i in indices)
        input_ids = np.full((len(indices), length), tokenizer.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(indices), length), dtype=np.int64)
        for row, i in enumerate(indices):
            ids = flat[i][1].input_ids
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(ids)] = 1
        batch_logits = tagger.logits(input_ids, attention_mask)
        for row, i in enumerate(indices):
            logits[i] = batch_logits[row]

    results: list[FloatArray] = []
    position = 0
    for words, paragraph_windows in zip(paragraphs, windows, strict=True):
        paragraph_logits = [
            logit
            for logit in logits[position : position + len(paragraph_windows)]
            if logit is not None
        ]
        position += len(paragraph_windows)
        results.append(
            word_probabilities(paragraph_windows, paragraph_logits, word_count=len(words))
        )
    return results
