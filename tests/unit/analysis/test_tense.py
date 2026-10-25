import pytest

from contcheck.analysis.models import TenseHistogram, Token
from contcheck.analysis.tense import tense_histogram


def verb(tag: str, *, finite: bool = True) -> Token:
    return Token("x", tag, "x", space_after=True, sentence_start=False, finite=finite)


def test_shares_of_finite_verbs() -> None:
    tokens = [
        verb("VBD"),
        verb("VBD"),
        verb("VBZ"),
        verb("MD"),
        verb("VB"),
        verb("NN", finite=False),
    ]

    histogram = tense_histogram(tokens)

    assert histogram == TenseHistogram(0.4, 0.2, 0.2, 0.2)
    assert sum(
        (histogram.past, histogram.present, histogram.modal, histogram.other)
    ) == pytest.approx(1)


def test_no_finite_verbs_gives_zeros() -> None:
    assert tense_histogram([verb("VBG", finite=False)]) == TenseHistogram(0.0, 0.0, 0.0, 0.0)
