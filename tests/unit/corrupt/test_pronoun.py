import random

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.pronoun import PronounCorruptor
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import DEFAULT_LEXICON, FakeAnalyzer

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)
LEXICON = {
    **DEFAULT_LEXICON,
    "me": ("PRP", "I"),
    "him": ("PRP", "he"),
    "her": ("PRP$", "her"),
    "myself": ("PRP", "myself"),
    "who": ("WP", "who"),
    "saw": ("VBD", "see"),
    "did": ("VBD", "do"),
    "man": ("NN", "man"),
    "was": ("VBD", "be"),
    "not": ("RB", "not"),
    "us": ("NNP", "US"),
}
ANALYZER = FakeAnalyzer(LEXICON)


def context(text: str) -> CorruptionContext:
    return CorruptionContext(ANALYZER.analyze(text).tokens, NO_TENSE)


def corrupt(text: str, rng: random.Random) -> Edit | None:
    return PronounCorruptor().corrupt(context(text), rng)


def test_swaps_subject_for_object_case(rng: random.Random) -> None:
    assert corrupt("Dogs saw he.", rng) == Edit(
        2, 3, ErrorKind.PRONOUN_FORM, Operation.REPLACEMENT, ("he",), ("him",)
    )


def test_swaps_a_possessive_for_the_subject_form(rng: random.Random) -> None:
    assert corrupt("Dogs saw her dog.", rng) == Edit(
        2, 3, ErrorKind.PRONOUN_FORM, Operation.REPLACEMENT, ("her",), ("she",)
    )


def test_misuses_a_reflexive(rng: random.Random) -> None:
    assert corrupt("Dogs saw myself.", rng) == Edit(
        2, 3, ErrorKind.PRONOUN_FORM, Operation.REPLACEMENT, ("myself",), ("me",)
    )


def test_object_me_becomes_i_or_myself() -> None:
    ctx = context("Dogs saw me.")

    news = {
        e.new for seed in range(20) if (e := PronounCorruptor().corrupt(ctx, random.Random(seed)))
    }

    assert news == {("I",), ("myself",)}


def test_i_is_a_site_only_at_a_sentence_start(rng: random.Random) -> None:
    assert corrupt("Dogs and I walk.", rng) is None
    assert corrupt("I walk.", rng) == Edit(
        0, 1, ErrorKind.PRONOUN_FORM, Operation.REPLACEMENT, ("I",), ("me",)
    )


def test_skips_pronouns_after_than_and_other_tags(rng: random.Random) -> None:
    assert corrupt("Dogs are big than he. The US is big.", rng) is None


def test_who_becomes_whom_only_as_a_relative_subject(rng: random.Random) -> None:
    assert corrupt("Who did walk?", rng) is None
    assert corrupt("The man who walks.", rng) == Edit(
        2, 3, ErrorKind.PRONOUN_FORM, Operation.REPLACEMENT, ("who",), ("whom",)
    )


@pytest.mark.parametrize("text", ["It was me.", "It was not me."])
def test_me_after_be_never_becomes_i(text: str) -> None:
    # "It was I" is the formal standard, so only the reflexive is a clear error.
    ctx = context(text)

    news = {
        e.new for seed in range(20) if (e := PronounCorruptor().corrupt(ctx, random.Random(seed)))
    }

    assert news == {("myself",)}
