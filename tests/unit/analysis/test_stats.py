import pytest

from contcheck.analysis.models import DocStats
from contcheck.analysis.stats import doc_stats
from tests.conftest import FakeAnalyzer


def test_counts_sentences_first_person_and_quotes(analyzer: FakeAnalyzer) -> None:
    paragraphs = [
        analyzer.analyze('I went home. She said " we walk " .'),
        analyzer.analyze("Dogs."),
    ]

    stats = doc_stats(paragraphs)

    assert stats.sentence_count == 3
    assert stats.mean_sentence_length == pytest.approx(8 / 3)
    assert stats.first_person_share == pytest.approx(2 / 8)
    assert stats.quoted_share == pytest.approx(2 / 8)


def test_empty_document() -> None:
    assert doc_stats([]) == DocStats(0, 0.0, 0.0, 0.0)
