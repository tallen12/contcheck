import pytest

from contcheck.taxonomy.kinds import ErrorKind
from contcheck.taxonomy.labels import (
    LABEL2ID,
    LABELS,
    OUTSIDE,
    Outside,
    Prefix,
    Tagged,
    format_label,
    parse_label,
)


def test_label_set_is_outside_plus_begin_and_inside_per_kind() -> None:
    assert len(LABELS) == 39
    assert LABELS[0] == OUTSIDE
    assert LABELS[1:3] == ("B-Spelling", "I-Spelling")


def test_label_ids_round_trip() -> None:
    assert all(LABELS[LABEL2ID[label]] == label for label in LABELS)
    assert sorted(LABEL2ID.values()) == list(range(len(LABELS)))


@pytest.mark.parametrize("kind", list(ErrorKind))
@pytest.mark.parametrize("prefix", list(Prefix))
def test_every_kind_round_trips_through_labels(prefix: Prefix, kind: ErrorKind) -> None:
    text = format_label(Tagged(prefix, kind))

    assert text in LABEL2ID
    assert parse_label(text) == Tagged(prefix, kind)


def test_outside_round_trips() -> None:
    assert parse_label(format_label(Outside())) == Outside()


@pytest.mark.parametrize("text", ["", "B", "X-Spelling", "B-Nope", "b-Spelling"])
def test_unknown_labels_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="unknown label"):
        parse_label(text)
