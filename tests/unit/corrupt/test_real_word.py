import random

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.real_word import RealWordCorruptor, load_confusion_sets, parse_confusion_sets
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)


def test_parse_maps_each_word_to_the_rest_of_its_set() -> None:
    confusions = parse_confusion_sets(["# comment", "", "Then than", "cite site sight  # note"])

    assert confusions["then"] == ("than",)
    assert confusions["site"] == ("cite", "sight")


@pytest.mark.parametrize("lines", [["a a"], ["solo"], ["a b", "b c"]])
def test_parse_rejects_bad_sets(lines: list[str]) -> None:
    with pytest.raises(ValueError, match="confusion set"):
        parse_confusion_sets(lines)


def test_bundled_sets_load_without_contractions() -> None:
    confusions = load_confusion_sets()

    assert "their" in confusions
    assert len(confusions) > 50
    assert not any("'" in word for word in confusions)


def test_corruptor_swaps_within_a_set(analyzer: FakeAnalyzer) -> None:
    corruptor = RealWordCorruptor(parse_confusion_sets(["then than", "form from"]))
    context = CorruptionContext(analyzer.analyze("Then I came from home.").tokens, NO_TENSE)

    edits = {corruptor.corrupt(context, random.Random(seed)) for seed in range(20)}

    assert {(e.orig, e.new) for e in edits if e is not None} == {
        (("Then",), ("than",)),
        (("from",), ("form",)),
    }
    assert all(e is not None and e.kind is ErrorKind.REAL_WORD for e in edits)
    assert all(e is not None and e.op is Operation.REPLACEMENT for e in edits)


def test_no_confusable_word_means_no_edit(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    corruptor = RealWordCorruptor(parse_confusion_sets(["then than"]))
    context = CorruptionContext(analyzer.analyze("I walk.").tokens, NO_TENSE)

    assert corruptor.corrupt(context, rng) is None
