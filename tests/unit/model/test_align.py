import pytest

from contcheck.corpus.record import Record, RecordSource, Split
from contcheck.model.align import (
    align_labels,
    make_windows,
    record_windows,
    window_starts,
)
from contcheck.model.models import IGNORE_INDEX, Encoding
from contcheck.taxonomy.labels import LABEL2ID
from tests.conftest import FakeTokenizer

X = IGNORE_INDEX


def test_only_the_first_subword_of_a_word_is_labeled() -> None:
    encoding = Encoding((10, 11, 12, 13, 14), (0, 0, 1, 2, 2))

    assert align_labels(encoding, [7, 8, 9]) == (7, X, 8, 9, X)


@pytest.mark.parametrize(
    ("length", "starts"),
    [(3, [0]), (4, [0]), (10, [0, 3, 6]), (9, [0, 3, 5]), (5, [0, 1])],
)
def test_window_starts_cover_every_token(length: int, starts: list[int]) -> None:
    assert window_starts(length, size=4, stride=3) == starts
    covered = {i for start in starts for i in range(start, min(start + 4, length))}
    assert covered == set(range(length))


def test_windows_add_special_tokens_and_keep_offsets(tokenizer: FakeTokenizer) -> None:
    encoding = Encoding(tuple(range(20, 30)), tuple(range(10)))
    labels = tuple(range(10))

    windows = make_windows(encoding, labels, tokenizer, doc_id="d", window=6, stride=3)

    assert [w.start for w in windows] == [0, 3, 6]
    first = windows[0]
    assert first.input_ids == (tokenizer.cls_id, 20, 21, 22, 23, tokenizer.sep_id)
    assert first.word_ids == (None, 0, 1, 2, 3, None)
    assert first.labels == (X, 0, 1, 2, 3, X)
    assert all(len(w.input_ids) <= 6 and w.doc_id == "d" for w in windows)


def test_record_windows_use_label_ids(tokenizer: FakeTokenizer) -> None:
    record = Record(
        doc_id="d",
        paragraph=0,
        variant=0,
        split=Split.TRAIN,
        source=RecordSource.RULE,
        clean="I went home.",
        noisy="I go home.",
        words=("I", "go", "home", "."),
        labels=("O", "B-VerbTense", "O", "O"),
        edits=(),
    )

    (window,) = record_windows(record, tokenizer, window=32, stride=16)

    assert window.word_ids == (None, 0, 1, 2, 2, 3, None)
    assert window.labels == (X, 0, LABEL2ID["B-VerbTense"], 0, X, 0, X)
