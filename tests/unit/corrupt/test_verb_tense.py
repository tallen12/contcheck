import random

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.verb_tense import VerbTenseCorruptor
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer, FakeInflector

PAST = TenseHistogram(0.9, 0.1, 0.0, 0.0)
PRESENT = TenseHistogram(0.1, 0.9, 0.0, 0.0)
MIXED = TenseHistogram(0.5, 0.5, 0.0, 0.0)


def edits(
    analyzer: FakeAnalyzer, inflector: FakeInflector, text: str, tense: TenseHistogram
) -> set[tuple[str, str]]:
    corruptor = VerbTenseCorruptor(inflector)
    context = CorruptionContext(analyzer.analyze(text).tokens, tense)
    found: set[tuple[str, str]] = set()
    for seed in range(20):
        edit = corruptor.corrupt(context, random.Random(seed))
        if edit is not None:
            assert edit.kind is ErrorKind.VERB_TENSE
            assert edit.op is Operation.REPLACEMENT
            found.add((edit.orig[0], edit.new[0]))
    return found


def test_past_documents_get_an_agreeing_present_verb(
    analyzer: FakeAnalyzer, inflector: FakeInflector
) -> None:
    found = edits(analyzer, inflector, "She walked home. They walked home. I was there.", PAST)

    assert found == {("walked", "walks"), ("walked", "walk"), ("was", "am")}


def test_present_documents_get_a_past_verb(
    analyzer: FakeAnalyzer, inflector: FakeInflector
) -> None:
    found = edits(analyzer, inflector, "She walks home. They are here.", PRESENT)

    assert found == {("walks", "walked"), ("are", "were")}


def test_mixed_documents_are_left_alone(analyzer: FakeAnalyzer, inflector: FakeInflector) -> None:
    assert edits(analyzer, inflector, "She walked home.", MIXED) == set()


def test_unknown_subjects_and_forms_are_skipped(
    analyzer: FakeAnalyzer, inflector: FakeInflector
) -> None:
    assert edits(analyzer, inflector, "The big walked. She flew.", PAST) == set()
