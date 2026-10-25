import random
from collections.abc import Mapping

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.adjective_adverb import (
    AdjectiveAdverbCorruptor,
    adjective_candidates,
    adverb_candidates,
)
from contcheck.corrupt.context import CorruptionContext
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer

ANY_TENSE = TenseHistogram(0.5, 0.5, 0.0, 0.0)

LEXICON: Mapping[str, tuple[str, str]] = {
    "she": ("PRP", "she"),
    "it": ("PRP", "it"),
    "a": ("DT", "a"),
    "the": ("DT", "the"),
    "and": ("CC", "and"),
    "spoke": ("VBD", "speak"),
    "ran": ("VBD", "run"),
    "opened": ("VBD", "open"),
    "did": ("VBD", "do"),
    "was": ("VBD", "be"),
    "is": ("VBZ", "be"),
    "looks": ("VBZ", "look"),
    "carefully": ("RB", "carefully"),
    "quickly": ("RB", "quickly"),
    "happily": ("RB", "happily"),
    "really": ("RB", "really"),
    "hardly": ("RB", "hardly"),
    "well": ("RB", "well"),
    "careful": ("JJ", "careful"),
    "cold": ("JJ", "cold"),
    "quiet": ("JJ", "quiet"),
    "good": ("JJ", "good"),
    "kind": ("JJ", "kind"),
    "several": ("JJ", "several"),
    "driver": ("NN", "driver"),
    "door": ("NN", "door"),
    "test": ("NN", "test"),
    "men": ("NNS", "man"),
}


class FakeLexicon:
    """A ``WordClassLexicon`` over two word lists."""

    def __init__(self, adjectives: set[str], adverbs: set[str]) -> None:
        self._adjectives = adjectives
        self._adverbs = adverbs

    def is_adjective(self, word: str) -> bool:
        return word.lower() in self._adjectives

    def is_adverb(self, word: str) -> bool:
        return word.lower() in self._adverbs


ADJECTIVES = {"careful", "quick", "happy", "real", "cold", "quiet", "good", "kind", "kindly"}
ADVERBS = {"carefully", "quickly", "quick", "happily", "really", "coldly", "quietly", "well"}
ADVERBS |= {"kindly", "hardly", "severally"}


@pytest.fixture
def analyzer() -> FakeAnalyzer:
    return FakeAnalyzer(LEXICON)


def swaps(analyzer: FakeAnalyzer, text: str) -> set[tuple[str, str]]:
    corruptor = AdjectiveAdverbCorruptor(FakeLexicon(ADJECTIVES | {"hard"}, ADVERBS))
    context = CorruptionContext(analyzer.analyze(text).tokens, ANY_TENSE)
    found: set[tuple[str, str]] = set()
    for seed in range(30):
        edit = corruptor.corrupt(context, random.Random(seed))
        if edit is not None:
            assert edit.kind is ErrorKind.ADJECTIVE_ADVERB
            assert edit.op is Operation.REPLACEMENT
            found.add((edit.orig[0], edit.new[0]))
    return found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("She spoke carefully.", {("carefully", "careful")}),
        ("She carefully opened the door.", {("carefully", "careful")}),
        ("She ran happily.", {("happily", "happy")}),
        ("She did well.", {("well", "good")}),
        ("It was really cold.", {("really", "real"), ("cold", "coldly")}),
        ("A careful driver.", {("careful", "carefully")}),
        ("A good test.", {("good", "well")}),
        ("It is quiet and cold.", {("quiet", "quietly")}),
    ],
)
def test_words_swap_where_the_slot_rules_them_out(
    analyzer: FakeAnalyzer, text: str, expected: set[tuple[str, str]]
) -> None:
    assert swaps(analyzer, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "She ran quickly.",  # "quick" is a flat adverb too
        "She hardly spoke.",  # "hardly" is not the adverb of "hard"
        "A kind driver.",  # "kindly" is an adjective too
        "Several men.",  # quantifier, not a manner word
        "She looks carefully.",  # complement of a linking verb
        "It is good.",  # "is well" can be right
        "The careful.",  # adjective not before a noun or ending a clause
    ],
)
def test_ambiguous_or_safe_slots_are_skipped(analyzer: FakeAnalyzer, text: str) -> None:
    assert swaps(analyzer, text) == set()


def test_derivation_rules() -> None:
    assert adverb_candidates("happy")[0] == "happily"
    assert adverb_candidates("simple")[0] == "simply"
    assert adverb_candidates("basic")[0] == "basically"
    assert adverb_candidates("full")[0] == "fully"
    assert adverb_candidates("true")[0] == "truly"
    assert adverb_candidates("quiet") == ("quietly",)
    assert adverb_candidates("lovely") == ()
    assert adjective_candidates("basically")[0] == "basic"
    assert adjective_candidates("happily")[0] == "happy"
    assert "full" in adjective_candidates("fully")
    assert "true" in adjective_candidates("truly")
    assert "simple" in adjective_candidates("simply")
    assert adjective_candidates("quickly")[0] == "quick"
    assert adjective_candidates("fast") == ()
