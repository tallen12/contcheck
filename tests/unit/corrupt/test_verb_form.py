import random
from collections.abc import Mapping

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.verb_form import Governor, VerbFormCorruptor, governor_of
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer, FakeInflector

ANY_TENSE = TenseHistogram(0.5, 0.5, 0.0, 0.0)

LEXICON: Mapping[str, tuple[str, str]] = {
    "she": ("PRP", "she"),
    "it": ("PRP", "it"),
    "the": ("DT", "the"),
    "has": ("VBZ", "have"),
    "is": ("VBZ", "be"),
    "can": ("MD", "can"),
    "to": ("TO", "to"),
    "did": ("VBD", "do"),
    "not": ("RB", "not"),
    "wants": ("VBZ", "want"),
    "gone": ("VBN", "go"),
    "going": ("VBG", "go"),
    "go": ("VB", "go"),
    "swim": ("VB", "swim"),
    "leave": ("VB", "leave"),
    "saw": ("VBD", "see"),
    "came": ("VBD", "come"),
    "walked": ("VBN", "walk"),
    "got": ("VBN", "get"),
}

INFLECTIONS: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("go", "VB"): ("go",),
    ("go", "VBD"): ("went",),
    ("go", "VBN"): ("gone",),
    ("go", "VBG"): ("going",),
    ("swim", "VB"): ("swim",),
    ("swim", "VBG"): ("swimming",),
    ("swim", "VBN"): ("swum",),
    ("leave", "VB"): ("leave",),
    ("leave", "VBG"): ("leaving",),
    ("leave", "VBN"): ("left",),
    ("see", "VB"): ("see",),
    ("see", "VBD"): ("saw",),
    ("see", "VBN"): ("seen",),
    ("come", "VB"): ("come",),
    ("come", "VBD"): ("came",),
    ("come", "VBN"): ("come",),
    ("walk", "VB"): ("walk",),
    ("walk", "VBD"): ("walked",),
    ("walk", "VBN"): ("walked",),
    ("get", "VB"): ("get",),
    ("get", "VBD"): ("got",),
    ("get", "VBN"): ("gotten", "got"),
}


@pytest.fixture
def analyzer() -> FakeAnalyzer:
    return FakeAnalyzer(LEXICON)


@pytest.fixture
def inflector() -> FakeInflector:
    return FakeInflector(INFLECTIONS)


def edits(analyzer: FakeAnalyzer, inflector: FakeInflector, text: str) -> set[tuple[str, str]]:
    corruptor = VerbFormCorruptor(inflector)
    context = CorruptionContext(analyzer.analyze(text).tokens, ANY_TENSE)
    found: set[tuple[str, str]] = set()
    for seed in range(30):
        edit = corruptor.corrupt(context, random.Random(seed))
        if edit is not None:
            assert edit.kind is ErrorKind.VERB_FORM
            assert edit.op is Operation.REPLACEMENT
            found.add((edit.orig[0], edit.new[0]))
    return found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("She has gone.", {("gone", "went"), ("gone", "go")}),
        ("She has not gone.", {("gone", "went"), ("gone", "go")}),
        ("She is going.", {("going", "go")}),
        ("She can swim.", {("swim", "swimming"), ("swim", "swum")}),
        ("She wants to leave.", {("leave", "leaving"), ("leave", "left")}),
        ("She did not go.", {("go", "went"), ("go", "going")}),
        ("She saw it.", {("saw", "seen")}),
    ],
)
def test_verbs_get_a_form_their_slot_rules_out(
    analyzer: FakeAnalyzer, inflector: FakeInflector, text: str, expected: set[tuple[str, str]]
) -> None:
    assert edits(analyzer, inflector, text) == expected


def test_forms_that_are_also_valid_in_the_slot_are_skipped(
    analyzer: FakeAnalyzer, inflector: FakeInflector
) -> None:
    assert edits(analyzer, inflector, "She has walked.") == {("walked", "walk")}
    assert edits(analyzer, inflector, "She has got it.") == {("got", "get")}
    assert edits(analyzer, inflector, "She came.") == set()


def test_participles_without_an_auxiliary_are_left_alone(
    analyzer: FakeAnalyzer, inflector: FakeInflector
) -> None:
    assert edits(analyzer, inflector, "The gone.") == set()


def test_governor_looks_back_past_adverbs_only(analyzer: FakeAnalyzer) -> None:
    def governor(text: str, verb: int) -> Governor:
        return governor_of(CorruptionContext(analyzer.analyze(text).tokens, ANY_TENSE), verb)

    assert governor("She has not gone.", 3) is Governor.HAVE
    assert governor("She did not go.", 3) is Governor.DO
    assert governor("She wants to leave.", 3) is Governor.TO
    assert governor("She can swim.", 2) is Governor.MODAL
    assert governor("She saw it.", 1) is Governor.NONE
    assert governor("Not gone.", 1) is Governor.NONE
