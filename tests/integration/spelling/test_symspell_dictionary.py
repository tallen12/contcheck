from contcheck.spelling.symspell_dictionary import load_symspell_dictionary


def test_bundled_frequency_list() -> None:
    dictionary = load_symspell_dictionary()

    assert dictionary.contains("house")
    assert not dictionary.contains("hosue")
    assert dictionary.suggestions("hosue")[0].term == "house"
