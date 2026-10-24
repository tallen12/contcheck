from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from tests.conftest import FakeAnalyzer


def test_context_exposes_token_columns(analyzer: FakeAnalyzer) -> None:
    tokens = analyzer.analyze("I went home").tokens
    context = CorruptionContext(tokens, TenseHistogram(1.0, 0.0, 0.0, 0.0))

    assert context.words == ("I", "went", "home")
    assert context.tags == ("PRP", "VBD", "NN")
    assert context.lemmas == ("I", "go", "home")
