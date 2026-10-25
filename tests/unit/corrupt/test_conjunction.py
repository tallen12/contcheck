import random

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.conjunction import ConjunctionCorruptor
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import DEFAULT_LEXICON, FakeAnalyzer

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)
LEXICON = {
    **DEFAULT_LEXICON,
    "but": ("CC", "but"),
    "or": ("CC", "or"),
    "because": ("IN", "because"),
    "although": ("IN", "although"),
    "if": ("IN", "if"),
    "so": ("RB", "so"),
    "that": ("IN", "that"),
    "not": ("RB", "not"),
}
ANALYZER = FakeAnalyzer(LEXICON)


def context(text: str) -> CorruptionContext:
    return CorruptionContext(ANALYZER.analyze(text).tokens, NO_TENSE)


def corrupt(text: str, rng: random.Random) -> Edit | None:
    return ConjunctionCorruptor().corrupt(context(text), rng)


def test_or_becomes_nor_without_a_negation(rng: random.Random) -> None:
    assert corrupt("I walk or I go.", rng) == Edit(
        2, 3, ErrorKind.CONJUNCTION, Operation.REPLACEMENT, ("or",), ("nor",)
    )


def test_or_stays_after_a_negation(rng: random.Random) -> None:
    # "I do not walk nor go" is correct, so the swap would label good English.
    assert corrupt("I do not walk or go.", rng) is None


def test_a_concessive_becomes_despite_even_at_a_sentence_start(rng: random.Random) -> None:
    assert corrupt("Although I walk, I go.", rng) == Edit(
        0, 1, ErrorKind.CONJUNCTION, Operation.REPLACEMENT, ("Although",), ("despite",)
    )


@pytest.mark.parametrize(
    "text",
    [
        "I walk and I go.",  # and for or or but only changes the meaning
        "I walk but I go.",
        "I walk because I go.",  # because for although too
        "If I walk, I go.",  # if for unless too
        "I walk, so I go.",  # so for but or because too
        "Or I walk.",  # sentence-initial coordinator
    ],
)
def test_meaning_only_swaps_are_not_made(text: str, rng: random.Random) -> None:
    assert corrupt(text, rng) is None


def test_leaves_than_and_that_alone(rng: random.Random) -> None:
    assert corrupt("I know that dogs are big than apples.", rng) is None
