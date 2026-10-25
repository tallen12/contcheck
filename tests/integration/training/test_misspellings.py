from contcheck.spelling.symspell_dictionary import load_symspell_dictionary
from contcheck.training.misspellings import load_misspellings


def test_bundled_misspellings_are_non_words_of_known_words() -> None:
    dictionary = load_symspell_dictionary()

    for correct, misspellings in load_misspellings().items():
        assert dictionary.contains(correct), correct
        for misspelling in misspellings:
            assert not dictionary.contains(misspelling), misspelling
