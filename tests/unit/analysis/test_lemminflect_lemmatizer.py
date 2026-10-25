from contcheck.analysis.lemminflect_lemmatizer import LemmInflectLemmatizer


def test_open_class_words_have_their_lemmas() -> None:
    lemmatizer = LemmInflectLemmatizer()

    assert lemmatizer.lemmas("forms") == {"form"}
    assert lemmatizer.lemmas("Went") == {"go"}


def test_stop_words_are_their_own_lemma() -> None:
    assert LemmInflectLemmatizer().lemmas("than") == {"than"}


def test_unknown_words_have_no_lemma() -> None:
    assert LemmInflectLemmatizer().lemmas("ebay") == frozenset()
