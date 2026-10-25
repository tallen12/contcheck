from collections.abc import Mapping

from contcheck.analysis.lexicon import Lexicon


class TableLexicon:
    """A ``Lexicon`` over a ``(word, tag) -> lemmas`` table."""

    def __init__(self, table: Mapping[tuple[str, str], frozenset[str]]) -> None:
        self._table = table

    def lemmas(self, word: str, tag: str) -> frozenset[str]:
        none = frozenset[str]()
        return self._table.get((word, tag), none)


def test_lexicon_contract() -> None:
    lexicon: Lexicon = TableLexicon({("ran", "VBD"): frozenset({"run"})})

    assert lexicon.lemmas("ran", "VBD") == {"run"}
    assert lexicon.lemmas("ran", "NN") == frozenset()
