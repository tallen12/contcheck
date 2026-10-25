import pytest

from contcheck.corpus.gutenberg import BookSelection, GutenbergArticles
from tests.integration.corpus.conftest import first_articles, small_stream


@pytest.mark.datasets
def test_streams_modern_fiction_books() -> None:
    source = GutenbergArticles(small_stream, seed=0, selection=BookSelection())

    articles = first_articles(source, count=1)

    assert len(articles) == 1
    assert articles[0].text.count("\n\n") > 50
