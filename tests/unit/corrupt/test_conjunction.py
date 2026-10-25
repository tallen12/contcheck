import random

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.conjunction import COORDINATING, SUBORDINATING, ConjunctionCorruptor
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import DEFAULT_LEXICON, FakeAnalyzer

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)
LEXICON = {
    **DEFAULT_LEXICON,
    "but": ("CC", "but"),
    "because": ("IN", "because"),
    "so": ("RB", "so"),
    "that": ("IN", "that"),
}
ANALYZER = FakeAnalyzer(LEXICON)


def context(text: str) -> CorruptionContext:
    return CorruptionContext(ANALYZER.analyze(text).tokens, NO_TENSE)


def corrupt(text: str, rng: random.Random) -> Edit | None:
    return ConjunctionCorruptor().corrupt(context(text), rng)


def test_swaps_a_coordinator(rng: random.Random) -> None:
    edit = corrupt("I walk and I go.", rng)

    assert edit is not None
    assert (edit.start, edit.kind, edit.op, edit.orig) == (
        2,
        ErrorKind.CONJUNCTION,
        Operation.REPLACEMENT,
        ("and",),
    )
    assert edit.new[0] in COORDINATING["and"]


def test_swaps_a_subordinator_even_at_a_sentence_start(rng: random.Random) -> None:
    edit = corrupt("Because I walk.", rng)

    assert edit is not None
    assert edit.orig == ("Because",)
    assert edit.new[0] in SUBORDINATING["because"]


def test_skips_a_sentence_initial_coordinator(rng: random.Random) -> None:
    assert corrupt("But I walk.", rng) is None


def test_so_is_a_conjunction_only_after_a_comma(rng: random.Random) -> None:
    assert corrupt("I walk so slowly.", rng) is None
    edit = corrupt("I walk, so I go.", rng)
    assert edit is not None
    assert edit.orig == ("so",)


def test_leaves_than_and_that_alone(rng: random.Random) -> None:
    assert corrupt("I know that dogs are big than apples.", rng) is None
