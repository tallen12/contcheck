import random
from collections.abc import Mapping

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.noun_number import NounNumberCorruptor, Number, marked_number
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer, FakeInflector

ANY_TENSE = TenseHistogram(0.5, 0.5, 0.0, 0.0)

LEXICON: Mapping[str, tuple[str, str]] = {
    "a": ("DT", "a"),
    "this": ("DT", "this"),
    "these": ("DT", "these"),
    "the": ("DT", "the"),
    "my": ("PRP$", "my"),
    "three": ("CD", "three"),
    "2": ("CD", "2"),
    "1990": ("CD", "1990"),
    "many": ("JJ", "many"),
    "big": ("JJ", "big"),
    "of": ("IN", "of"),
    "dog": ("NN", "dog"),
    "dogs": ("NNS", "dog"),
    "book": ("NN", "book"),
    "books": ("NNS", "book"),
    "children": ("NNS", "child"),
    "sheep": ("NNS", "sheep"),
    "clothes": ("NNS", "clothe"),
    "store": ("NN", "store"),
    "list": ("NN", "list"),
    "barks": ("VBZ", "bark"),
    "bark": ("VBP", "bark"),
    "was": ("VBD", "be"),
    "is": ("VBZ", "be"),
}

INFLECTIONS: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("dog", "NN"): ("dog",),
    ("dog", "NNS"): ("dogs", "dog"),
    ("book", "NN"): ("book",),
    ("book", "NNS"): ("books",),
    ("child", "NN"): ("child",),
    ("child", "NNS"): ("children",),
    ("sheep", "NN"): ("sheep",),
    ("sheep", "NNS"): ("sheep",),
    ("clothe", "NN"): ("clothe",),
    ("clothe", "NNS"): ("clothes",),
}


@pytest.fixture
def analyzer() -> FakeAnalyzer:
    return FakeAnalyzer(LEXICON)


@pytest.fixture
def inflector() -> FakeInflector:
    return FakeInflector(INFLECTIONS)


def flips(analyzer: FakeAnalyzer, inflector: FakeInflector, text: str) -> set[tuple[str, str]]:
    corruptor = NounNumberCorruptor(inflector)
    context = CorruptionContext(analyzer.analyze(text).tokens, ANY_TENSE)
    found: set[tuple[str, str]] = set()
    for seed in range(30):
        edit = corruptor.corrupt(context, random.Random(seed))
        if edit is not None:
            assert edit.kind is ErrorKind.NOUN_NUMBER
            assert edit.op is Operation.REPLACEMENT
            found.add((edit.orig[0], edit.new[0]))
    return found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("This book.", {("book", "books")}),
        ("A big dog.", {("dog", "dogs")}),
        ("These big books.", {("books", "book")}),
        ("Three children.", {("children", "child")}),
        ("Many dogs.", {("dogs", "dog")}),
        ("Only 2 dogs.", {("dogs", "dog")}),
        ("The dog barks.", {("dog", "dogs")}),
        ("My dogs bark.", {("dogs", "dog")}),
        ("The book was here.", {("book", "books")}),
    ],
)
def test_nouns_flip_against_their_marked_number(
    analyzer: FakeAnalyzer, inflector: FakeInflector, text: str, expected: set[tuple[str, str]]
) -> None:
    assert flips(analyzer, inflector, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The dog.",  # no marker
        "In 1990 dogs bark.",  # a year is no numeral marker
        "The list of the dogs is long.",  # the noun after "of" is not the subject
        "This dog store.",  # "dog" is not the head
        "These sheep.",  # invariant plural
        "These clothes.",  # plural only
        "This Book.",  # capitalized, maybe a name
    ],
)
def test_unmarked_or_unsafe_nouns_are_skipped(
    analyzer: FakeAnalyzer, inflector: FakeInflector, text: str
) -> None:
    assert flips(analyzer, inflector, text) == set()


def test_marked_number(analyzer: FakeAnalyzer) -> None:
    def number(text: str, noun: int) -> Number | None:
        return marked_number(CorruptionContext(analyzer.analyze(text).tokens, ANY_TENSE), noun)

    assert number("A big dog.", 2) is Number.SINGULAR
    assert number("Many dogs.", 1) is Number.PLURAL
    assert number("My dogs bark.", 1) is Number.PLURAL
    assert number("The dog.", 1) is None
