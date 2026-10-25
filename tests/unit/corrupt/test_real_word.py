import random
from collections import Counter

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.confusables import Confusion, ConfusionMap, ConfusionSource, hand_written_map
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.real_word import (
    RealWordCorruptor,
    load_confusion_sets,
    parse_confusion_sets,
    weighted_index,
)
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
    corruptor = RealWordCorruptor(
        hand_written_map(parse_confusion_sets(["then than", "form from"]))
    )
    context = CorruptionContext(analyzer.analyze("Then I came from home.").tokens, NO_TENSE)

    edits = {corruptor.corrupt(context, random.Random(seed)) for seed in range(20)}

    assert {(e.orig, e.new) for e in edits if e is not None} == {
        # Casing is the applier's job: it turns "than" into "Than" at the sentence start.
        (("Then",), ("than",)),
        (("from",), ("form",)),
    }
    assert all(e is not None and e.kind is ErrorKind.REAL_WORD for e in edits)
    assert all(e is not None and e.op is Operation.REPLACEMENT for e in edits)


def test_no_confusable_word_means_no_edit(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    corruptor = RealWordCorruptor(hand_written_map(parse_confusion_sets(["then than"])))
    context = CorruptionContext(analyzer.analyze("I walk.").tokens, NO_TENSE)

    assert corruptor.corrupt(context, rng) is None


def lookalike(word: str, alternative: str, weight: float) -> Confusion:
    return Confusion(word, alternative, weight, ConfusionSource.LOOK_ALIKE)


def test_corruptor_follows_the_weights(analyzer: FakeAnalyzer) -> None:
    confusions = ConfusionMap.of(
        [
            lookalike("then", "than", 99.0),
            lookalike("then", "them", 1.0),
            lookalike("from", "form", 1.0),
        ]
    )
    corruptor = RealWordCorruptor(confusions)
    context = CorruptionContext(analyzer.analyze("Then I came from home.").tokens, NO_TENSE)

    edits = [corruptor.corrupt(context, random.Random(seed)) for seed in range(200)]
    news = Counter(e.new[0] for e in edits if e is not None)

    assert news["than"] > 10 * news["them"]
    assert news["than"] > 10 * news["form"]


def test_repeated_words_are_not_favoured(analyzer: FakeAnalyzer) -> None:
    confusions = ConfusionMap.of([lookalike("the", "she", 1.0), lookalike("from", "form", 1.0)])
    corruptor = RealWordCorruptor(confusions)
    text = "The dog went from the park to the house and the store."
    context = CorruptionContext(analyzer.analyze(text).tokens, NO_TENSE)

    edits = [corruptor.corrupt(context, random.Random(seed)) for seed in range(400)]
    news = Counter(e.new[0].lower() for e in edits if e is not None)

    assert 0.35 < news["form"] / sum(news.values()) < 0.65


def test_weighted_index_draws_in_proportion() -> None:
    rng = random.Random(3)

    counts = Counter(weighted_index([1.0, 0.0, 3.0], rng) for _ in range(4000))

    assert counts[1] == 0
    assert 2.5 < counts[2] / counts[0] < 3.5
