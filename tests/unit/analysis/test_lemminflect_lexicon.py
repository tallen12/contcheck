from contcheck.analysis.lemminflect_lexicon import LemmLexicon


def test_known_adjectives_and_adverbs() -> None:
    lexicon = LemmLexicon()

    assert lexicon.is_adjective("careful")
    assert lexicon.is_adverb("Carefully")
    assert not lexicon.is_adjective("on")
    assert not lexicon.is_adverb("greenly")
