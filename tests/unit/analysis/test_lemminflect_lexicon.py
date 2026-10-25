import pytest

from contcheck.analysis.lemminflect_lexicon import LemmInflectLexicon


@pytest.mark.parametrize(
    ("word", "tag", "lemmas"),
    [
        ("ran", "VBD", {"run"}),
        ("walked", "VBN", {"walk"}),
        ("dogs", "NNS", {"dog"}),
        ("happier", "JJR", {"happy"}),
        ("quickly", "RB", {"quickly"}),
        ("better", "JJR", {"good", "well"}),
    ],
)
def test_known_forms_have_their_lemmas(word: str, tag: str, lemmas: set[str]) -> None:
    assert LemmInflectLexicon().lemmas(word, tag) == lemmas


@pytest.mark.parametrize(
    ("word", "tag"),
    [
        ("ran", "NN"),  # a verb form, not a noun
        ("nationally", "NN"),  # lemminflect would inflect any lemma for any tag
        ("way", "RB"),
        ("xqzzy", "NN"),  # not a dictionary word
        ("the", "DT"),  # a tag lemminflect does not inflect
    ],
)
def test_wrong_tags_and_unknown_words_have_none(word: str, tag: str) -> None:
    assert LemmInflectLexicon().lemmas(word, tag) == frozenset()
