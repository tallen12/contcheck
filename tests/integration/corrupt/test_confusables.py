import pytest

from contcheck.analysis.lemminflect_lemmatizer import LemmInflectLemmatizer
from contcheck.corrupt.confusables import (
    ConfusionMap,
    ConfusionSource,
    build_confusion_map,
    load_homophone_groups,
)
from contcheck.corrupt.real_word import load_confusion_sets
from contcheck.spelling.symspell_dictionary import load_symspell_dictionary


@pytest.fixture(scope="module")
def confusions() -> ConfusionMap:
    dictionary = load_symspell_dictionary()
    return build_confusion_map(
        frequencies=dictionary.frequencies(),
        dictionary=dictionary,
        lemmatizer=LemmInflectLemmatizer(),
        confusion_sets=load_confusion_sets(),
        homophones=load_homophone_groups(),
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [("form", "from"), ("three", "there"), ("quiet", "quite"), ("piece", "peace")],
)
def test_known_confusions_are_found(confusions: ConfusionMap, a: str, b: str) -> None:
    assert frozenset({a, b}) in confusions.pairs()


def test_generated_pairs_include_look_alikes_and_homophones(confusions: ConfusionMap) -> None:
    sources = {c.pair: c.source for c in confusions}

    assert sources[frozenset({"three", "there"})] is ConfusionSource.LOOK_ALIKE
    assert sources[frozenset({"to", "two"})] is ConfusionSource.HOMOPHONE
    assert len(confusions.pairs()) > 10_000


@pytest.mark.parametrize(
    ("a", "b"),
    [("form", "forms"), ("walk", "walked"), ("use", "used"), ("color", "colour")],
)
def test_forms_of_one_word_are_not_confusions(confusions: ConfusionMap, a: str, b: str) -> None:
    assert frozenset({a, b}) not in confusions.pairs()


def test_no_word_carries_more_than_the_cap(confusions: ConfusionMap) -> None:
    assert max(confusions.weight(word) for word in confusions.by_word) <= 3.0 + 1e-9
