import spacy

from contcheck.analysis.spacy_analyzer import SpacyAnalyzer


def test_converts_tokens_and_drops_whitespace_tokens() -> None:
    # A blank pipeline tokenizes without tagging, so this runs without the model.
    analyzer = SpacyAnalyzer(spacy.blank("en"))

    analyzed = analyzer.analyze("Hi,  you.")

    assert analyzed.words == ("Hi", ",", "you", ".")
    assert [t.space_after for t in analyzed.tokens] == [False, True, False, False]
    assert analyzed.tokens[0].sentence_start
    assert not any(t.finite for t in analyzed.tokens)


def test_analyze_many_keeps_order() -> None:
    analyzer = SpacyAnalyzer(spacy.blank("en"))

    results = list(analyzer.analyze_many(["one", "two three"]))

    assert [r.words for r in results] == [("one",), ("two", "three")]
