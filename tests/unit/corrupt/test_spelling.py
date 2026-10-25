import random
from collections.abc import Callable

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.spelling import (
    KEYBOARD_TYPOS,
    NEIGHBOURS,
    REALISTIC_TYPOS,
    TYPOS,
    SpellingCorruptor,
    delete_letter,
    double_for_single,
    double_letter,
    phonetic_swap,
    silent_e,
    single_for_double,
    substitute_adjacent_key,
    swap_unstressed_vowel,
    transpose,
    transpose_vowels,
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


def outputs(typo: Callable[[str, Rng], str | None], word: str) -> set[str | None]:
    return {typo(word, random.Random(seed)) for seed in range(40)}


@pytest.mark.parametrize(
    ("typo", "word", "expected"),
    [
        (single_for_double, "occurred", {"occured", "ocurred"}),
        (single_for_double, "tomorrow", {"tomorow"}),
        (single_for_double, "accommodate", {"accomodate", "acommodate"}),
        (double_for_single, "necessary", {"neccessary"}),
        (swap_unstressed_vowel, "separate", {"seperate", "sepirate", "separete", "separite"}),
        (silent_e, "truly", {"truely"}),
        (silent_e, "argument", {"arguement"}),
        (silent_e, "excitement", {"excitment"}),
        (transpose_vowels, "because", {"becuase"}),
        (phonetic_swap, "independent", {"independant"}),
        (phonetic_swap, "publicly", {"publically"}),
    ],
)
def test_realistic_typos_produce_known_misspellings(
    typo: Callable[[str, Rng], str | None], word: str, expected: set[str]
) -> None:
    assert outputs(typo, word) == expected


@pytest.mark.parametrize(
    ("typo", "word"),
    [
        (single_for_double, "garden"),
        (double_for_single, "strength"),
        (swap_unstressed_vowel, "about"),
        (silent_e, "garden"),
        (transpose_vowels, "garden"),
    ],
)
def test_realistic_typos_without_a_site(typo: Callable[[str, Rng], str | None], word: str) -> None:
    assert outputs(typo, word) == {None}


def test_realistic_typos_are_the_majority_of_attempts() -> None:
    realistic = sum(typo in REALISTIC_TYPOS for typo in TYPOS)

    assert realistic == 2 * len(REALISTIC_TYPOS)
    assert realistic > 2 * len(KEYBOARD_TYPOS)
