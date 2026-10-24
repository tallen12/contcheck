import pytest

from contcheck.analysis.spacy_analyzer import SpacyAnalyzer, load_spacy_analyzer
from contcheck.analysis.tense import tense_histogram


@pytest.fixture(scope="module")
def analyzer() -> SpacyAnalyzer:
    return load_spacy_analyzer()


def test_tags_lemmas_and_finite_verbs(analyzer: SpacyAnalyzer) -> None:
    analyzed = analyzer.analyze("Yesterday she went to the store. She will buy apples.")
    by_word = {t.text: t for t in analyzed.tokens}

    assert by_word["went"].tag == "VBD"
    assert by_word["went"].lemma == "go"
    assert by_word["went"].finite
    assert by_word["will"].finite
    assert not by_word["buy"].finite
    assert by_word["apples"].tag == "NNS"
    assert [t.text for t in analyzed.tokens if t.sentence_start] == ["Yesterday", "She"]
    assert analyzed.text == "Yesterday she went to the store. She will buy apples."


def test_past_tense_story_is_mostly_past(analyzer: SpacyAnalyzer) -> None:
    story = (
        "The old man walked to the harbour every morning. He watched the boats and "
        "counted the gulls. One day a stranger arrived and asked him about the sea. "
        "They talked for hours, and the stranger left before sunset."
    )

    histogram = tense_histogram(analyzer.analyze(story).tokens)

    assert histogram.past > 0.8
