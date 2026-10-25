import pytest

from contcheck.taxonomy.bio import WordSpan, labels_to_spans, missing_anchor, spans_to_labels
from contcheck.taxonomy.kinds import ErrorKind


@pytest.mark.parametrize("kind", list(ErrorKind))
def test_spans_round_trip_through_labels_for_every_kind(kind: ErrorKind) -> None:
    spans = [WordSpan(0, 1, kind), WordSpan(2, 5, kind), WordSpan(5, 6, ErrorKind.SPELLING)]

    labels = spans_to_labels(spans, word_count=7)

    assert labels_to_spans(labels) == spans


def test_spans_to_labels_writes_begin_then_inside() -> None:
    labels = spans_to_labels([WordSpan(1, 3, ErrorKind.VERB_TENSE)], word_count=4)

    assert labels == ["O", "B-VerbTense", "I-VerbTense", "O"]


def test_spans_to_labels_rejects_overlap_and_overflow() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        spans_to_labels(
            [WordSpan(0, 2, ErrorKind.SPELLING), WordSpan(1, 3, ErrorKind.DETERMINER)],
            word_count=4,
        )
    with pytest.raises(ValueError, match="exceeds"):
        spans_to_labels([WordSpan(3, 5, ErrorKind.SPELLING)], word_count=4)


@pytest.mark.parametrize(("start", "end"), [(-1, 1), (2, 2), (3, 1)])
def test_word_span_rejects_empty_or_negative(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="invalid span"):
        WordSpan(start, end, ErrorKind.SPELLING)


def test_orphan_inside_starts_a_new_span() -> None:
    spans = labels_to_spans(["I-Spelling", "I-Spelling", "O", "I-Determiner"])

    assert spans == [WordSpan(0, 2, ErrorKind.SPELLING), WordSpan(3, 4, ErrorKind.DETERMINER)]


def test_inside_of_another_kind_starts_a_new_span() -> None:
    spans = labels_to_spans(["B-Spelling", "I-RealWord"])

    assert spans == [WordSpan(0, 1, ErrorKind.SPELLING), WordSpan(1, 2, ErrorKind.REAL_WORD)]


def test_adjacent_begins_are_separate_spans() -> None:
    spans = labels_to_spans(["B-Spelling", "B-Spelling"])

    assert spans == [WordSpan(0, 1, ErrorKind.SPELLING), WordSpan(1, 2, ErrorKind.SPELLING)]


def test_labels_to_spans_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        labels_to_spans(["O", "B-Nope"])


def test_missing_tags_the_word_after_the_gap() -> None:
    assert missing_anchor(0, word_count=3) == 0
    assert missing_anchor(2, word_count=3) == 2


def test_missing_at_the_end_tags_the_word_before() -> None:
    assert missing_anchor(3, word_count=3) == 2


@pytest.mark.parametrize(("gap", "count"), [(-1, 3), (4, 3), (0, 0)])
def test_missing_anchor_rejects_bad_gaps(gap: int, count: int) -> None:
    with pytest.raises(ValueError):
        missing_anchor(gap, word_count=count)


def test_spans_to_labels_rejects_negative_word_count() -> None:
    with pytest.raises(ValueError, match="negative word count"):
        spans_to_labels([], word_count=-1)


def test_the_last_two_gaps_share_an_anchor() -> None:
    assert missing_anchor(2, word_count=3) == missing_anchor(3, word_count=3)
