import pytest
from symspellpy import SymSpell

from contcheck.spelling.symspell_dictionary import SymSpellDictionary


@pytest.fixture
def dictionary() -> SymSpellDictionary:
    symspell = SymSpell(max_dictionary_edit_distance=2)
    for word, count in [("the", 100), ("they", 50), ("then", 40), ("house", 10)]:
        symspell.create_dictionary_entry(word, count)
    return SymSpellDictionary(symspell)


def test_membership_ignores_case(dictionary: SymSpellDictionary) -> None:
    assert dictionary.contains("The")
    assert not dictionary.contains("teh")


def test_suggestions_are_closest_then_most_frequent(dictionary: SymSpellDictionary) -> None:
    suggestions = dictionary.suggestions("Thex", max_distance=1)

    assert [s.term for s in suggestions] == ["the", "they", "then"]
    assert {s.distance for s in suggestions} == {1}


def test_no_suggestions_beyond_distance(dictionary: SymSpellDictionary) -> None:
    assert dictionary.suggestions("zzzzz", max_distance=2) == []
