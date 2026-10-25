"""From token logits to word probabilities to error spans.

Pure numpy, so inference can run from ONNX Runtime without torch.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from contcheck.infer.models import SpanError
from contcheck.model.models import IGNORE_INDEX, Window
from contcheck.taxonomy.bio import labels_to_spans
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.taxonomy.labels import LABEL2ID, LABELS, OUTSIDE

type FloatArray = npt.NDArray[np.float32]


def softmax(logits: FloatArray) -> FloatArray:
    """Probabilities over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def word_probabilities(
    windows: Sequence[Window], logits: Sequence[FloatArray], *, word_count: int
) -> FloatArray:
    """Label probabilities per word, from each word's first subword.

    Windows overlap, so a word can be scored more than once; the window where its first
    subword sits closest to the centre wins, since it sees the most context on both
    sides. A word's first subword is where the window's label is not ignored, so windows
    for inference need placeholder labels from :func:`contcheck.model.align.align_labels`.
    Words no window scored get probability 1 for ``O``.
    """
    probabilities = np.zeros((word_count, len(LABELS)), dtype=np.float32)
    probabilities[:, LABEL2ID[OUTSIDE]] = 1.0
    best_distance = np.full(word_count, np.inf)
    for window, window_logits in zip(windows, logits, strict=True):
        centre = (len(window.input_ids) - 1) / 2
        token_probabilities = softmax(window_logits[: len(window.input_ids)])
        for position, (word_id, label) in enumerate(
            zip(window.word_ids, window.labels, strict=True)
        ):
            if word_id is None or label == IGNORE_INDEX:
                continue
            distance = abs(position - centre)
            if distance < best_distance[word_id]:
                best_distance[word_id] = distance
                probabilities[word_id] = token_probabilities[position]
    return probabilities


def decode(
    probabilities: FloatArray,
    *,
    thresholds: Mapping[ErrorKind, float] | None = None,
    default_threshold: float = 0.0,
) -> list[SpanError]:
    """Spans from each word's most likely label, keeping those above their kind's threshold.

    Invalid BIO sequences are repaired as in :func:`labels_to_spans`. A span's probability
    is the mean probability of its predicted labels.
    """
    predicted = probabilities.argmax(axis=-1)
    spans = labels_to_spans([LABELS[int(i)] for i in predicted])
    errors: list[SpanError] = []
    for span in spans:
        rows = np.arange(span.start, span.end)
        probability = float(probabilities[rows, predicted[rows]].mean())
        threshold = (thresholds or {}).get(span.kind, default_threshold)
        if probability >= threshold:
            errors.append(SpanError(span.start, span.end, span.kind, probability))
    return errors
