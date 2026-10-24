import random

from contcheck.analysis.protocols import Inflector, TextAnalyzer
from contcheck.corrupt.protocols import Rng
from contcheck.spelling.protocols import Dictionary
from tests.conftest import FakeAnalyzer, FakeDictionary, FakeInflector, edit_distance


def test_fakes_implement_their_protocols(
    analyzer: FakeAnalyzer,
    inflector: FakeInflector,
    dictionary: FakeDictionary,
    rng: random.Random,
) -> None:
    implementations: tuple[TextAnalyzer, Inflector, Dictionary, Rng] = (
        analyzer,
        inflector,
        dictionary,
        rng,
    )

    assert len(implementations) == 4


def test_fake_analyzer_tags_from_the_lexicon_and_keeps_spacing(analyzer: FakeAnalyzer) -> None:
    analyzed = analyzer.analyze("Yesterday I went home. She walks.")

    assert analyzed.words == ("Yesterday", "I", "went", "home", ".", "She", "walks", ".")
    assert [t.tag for t in analyzed.tokens] == ["NN", "PRP", "VBD", "NN", ".", "PRP", "VBZ", "."]
    assert [t.lemma for t in analyzed.tokens][2] == "go"
    assert [t.sentence_start for t in analyzed.tokens] == [
        True, False, False, False, False, True, False, False,
    ]  # fmt: skip
    assert [t.finite for t in analyzed.tokens].count(True) == 2
    assert analyzed.text == "Yesterday I went home. She walks."


def test_fake_analyzer_analyzes_many_in_order(analyzer: FakeAnalyzer) -> None:
    results = list(analyzer.analyze_many(["a dog", "the dogs"]))

    assert [r.words for r in results] == [("a", "dog"), ("the", "dogs")]


def test_fake_inflector_returns_table_forms(inflector: FakeInflector) -> None:
    assert inflector.inflect("go", "VBZ") == ("goes",)
    assert inflector.inflect("fly", "VBZ") == ()


def test_fake_dictionary_looks_up_ignoring_case(dictionary: FakeDictionary) -> None:
    assert dictionary.contains("The")
    assert not dictionary.contains("teh")


def test_fake_dictionary_suggests_closest_then_most_frequent(dictionary: FakeDictionary) -> None:
    suggestions = dictionary.suggestions("teh", max_distance=2)

    assert suggestions[0].term == "the"
    assert all(s.distance <= 2 for s in suggestions)
    assert [s.distance for s in suggestions] == sorted(s.distance for s in suggestions)


def test_edit_distance() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("same", "same") == 0
