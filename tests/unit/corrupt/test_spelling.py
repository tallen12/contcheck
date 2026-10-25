import random

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.spelling import (
    NEIGHBOURS,
    TYPOS,
    SpellingCorruptor,
    delete_letter,
    double_letter,
    phonetic_swap,
    substitute_adjacent_key,
    transpose,
)
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer, FakeDictionary

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)


def test_keyboard_neighbours_are_adjacent_keys() -> None:
    assert set(NEIGHBOURS["s"]) == set("qweadzxc")
    assert "p" not in NEIGHBOURS["a"]


@pytest.mark.parametrize("seed", range(20))
def test_typos_keep_the_first_letter(seed: int) -> None:
    rng = random.Random(seed)
    for typo in TYPOS:
        result = typo("Garden", rng)
        if result is not None:
            assert result[0] == "G"
            assert result != "Garden" or typo is substitute_adjacent_key


def test_each_typo_operation() -> None:
    rng = random.Random(0)

    assert len(delete_letter("house", rng) or "") == 4
    assert len(double_letter("house", rng) or "") == 6
    assert sorted(transpose("house", rng) or "") == sorted("house")
    assert phonetic_swap("telephone", rng) == "telefone"
    assert phonetic_swap("dog", rng) is None


def test_adjacent_key_keeps_letter_case() -> None:
    result = substitute_adjacent_key("ABCD", random.Random(1))

    assert result is not None
    assert result.isupper()


def test_corruptor_misspells_a_known_word_into_a_non_word(
    analyzer: FakeAnalyzer, dictionary: FakeDictionary
) -> None:
    corruptor = SpellingCorruptor(dictionary)
    context = CorruptionContext(analyzer.analyze("The dog went to the garden.").tokens, NO_TENSE)

    for seed in range(30):
        edit = corruptor.corrupt(context, random.Random(seed))
        assert edit is not None
        assert edit.kind is ErrorKind.SPELLING
        assert edit.op is Operation.REPLACEMENT
        assert edit.orig in {("went",), ("garden",)}
        assert not dictionary.contains(edit.new[0])


def test_corruptor_skips_names_short_words_and_unknown_words(
    analyzer: FakeAnalyzer, dictionary: FakeDictionary
) -> None:
    corruptor = SpellingCorruptor(dictionary)
    context = CorruptionContext(analyzer.analyze("I saw London and Zorblax.").tokens, NO_TENSE)

    assert corruptor.corrupt(context, random.Random(0)) is None
