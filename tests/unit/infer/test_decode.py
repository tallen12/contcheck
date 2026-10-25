import subprocess
import sys

import numpy as np
import pytest

from contcheck.infer.decode import decode, softmax, word_probabilities
from contcheck.infer.models import SpanError
from contcheck.model.models import IGNORE_INDEX, Window
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.taxonomy.labels import LABEL2ID, LABELS

X = IGNORE_INDEX


def one_hot(rows: list[str], *, confidence: float = 0.9) -> np.ndarray:
    probabilities = np.full((len(rows), len(LABELS)), (1 - confidence) / (len(LABELS) - 1))
    for i, label in enumerate(rows):
        probabilities[i, LABEL2ID[label]] = confidence
    return probabilities.astype(np.float32)


def test_softmax_rows_sum_to_one() -> None:
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, 0.0]], dtype=np.float32))

    assert probabilities.sum(axis=-1) == pytest.approx([1.0, 1.0])
    assert probabilities[1, 0] == pytest.approx(1.0)


def test_word_probabilities_use_first_subwords() -> None:
    window = Window("d", (1, 10, 11, 12, 2), (None, 0, 0, 1, None), (X, 0, X, 0, X), start=0)
    logits = np.log(one_hot(["O", "B-Spelling", "O", "B-RealWord", "O"]) + 1e-9)

    probabilities = word_probabilities([window], [logits], word_count=2)

    assert probabilities.argmax(axis=-1).tolist() == [
        LABEL2ID["B-Spelling"],
        LABEL2ID["B-RealWord"],
    ]


def test_centre_most_window_wins() -> None:
    # Word 1 sits at the edge of the first window and the centre of the second.
    first = Window("d", (1, 10, 11, 2), (None, 0, 1, None), (X, 0, 0, X), start=0)
    second = Window("d", (1, 11, 2), (None, 1, None), (X, 0, X), start=1)
    first_logits = np.log(one_hot(["O", "O", "B-Spelling", "O"]) + 1e-9)
    second_logits = np.log(one_hot(["O", "B-RealWord", "O"]) + 1e-9)

    probabilities = word_probabilities([first, second], [first_logits, second_logits], word_count=3)

    assert LABELS[int(probabilities[1].argmax())] == "B-RealWord"
    assert LABELS[int(probabilities[2].argmax())] == "O"


def test_decode_groups_and_repairs_spans() -> None:
    probabilities = one_hot(["O", "I-Spelling", "I-Spelling", "O", "B-VerbTense"])

    errors = decode(probabilities)

    assert [(e.start, e.end, e.kind) for e in errors] == [
        (1, 3, ErrorKind.SPELLING),
        (4, 5, ErrorKind.VERB_TENSE),
    ]
    assert all(isinstance(e, SpanError) and e.probability == pytest.approx(0.9) for e in errors)


def test_decode_applies_per_kind_thresholds() -> None:
    probabilities = one_hot(["B-Spelling", "O", "B-VerbTense"], confidence=0.6)

    errors = decode(probabilities, thresholds={ErrorKind.SPELLING: 0.7}, default_threshold=0.5)

    assert [e.kind for e in errors] == [ErrorKind.VERB_TENSE]


def test_decoding_does_not_import_torch() -> None:
    code = "import sys, contcheck.infer.decode; sys.exit('torch' in sys.modules)"

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
