import random

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.agreement import AgreementCorruptor
from contcheck.corrupt.context import CorruptionContext
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer, FakeInflector

ANY_TENSE = TenseHistogram(0.5, 0.5, 0.0, 0.0)


def swaps(analyzer: FakeAnalyzer, inflector: FakeInflector, text: str) -> set[tuple[str, str]]:
    corruptor = AgreementCorruptor(inflector)
    context = CorruptionContext(analyzer.analyze(text).tokens, ANY_TENSE)
    found: set[tuple[str, str]] = set()
    for seed in range(30):
        edit = corruptor.corrupt(context, random.Random(seed))
        if edit is not None:
            assert edit.kind is ErrorKind.SUBJECT_VERB_AGREEMENT
            assert edit.op is Operation.REPLACEMENT
            found.add((edit.orig[0], edit.new[0]))
    return found


def test_present_verbs_swap_person(analyzer: FakeAnalyzer, inflector: FakeInflector) -> None:
    assert swaps(analyzer, inflector, "She walks. They walk.") == {
        ("walks", "walk"),
        ("walk", "walks"),
    }


def test_be_swaps_in_present_and_past(analyzer: FakeAnalyzer, inflector: FakeInflector) -> None:
    assert swaps(analyzer, inflector, "It is here. They was there.") == {
        ("is", "are"),
        ("was", "were"),
    }


def test_past_and_modal_verbs_have_no_agreement(
    analyzer: FakeAnalyzer, inflector: FakeInflector
) -> None:
    assert swaps(analyzer, inflector, "She walked. They can.") == set()
