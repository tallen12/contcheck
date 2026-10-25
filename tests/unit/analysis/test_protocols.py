from contcheck.analysis.protocols import Inflector, TextAnalyzer
from tests.conftest import FakeAnalyzer, FakeInflector


def test_analyzer_contract(analyzer: FakeAnalyzer) -> None:
    implementation: TextAnalyzer = analyzer

    one = implementation.analyze("I walk.")
    many = list(implementation.analyze_many(["I walk.", "She walks."]))

    assert many[0] == one
    assert len(many) == 2


def test_inflector_contract(inflector: FakeInflector) -> None:
    implementation: Inflector = inflector

    assert implementation.inflect("walk", "VBD") == ("walked",)
    assert implementation.inflect("unknown", "VBD") == ()
