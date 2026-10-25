import pytest

from contcheck.corpus.fineweb import FineWebEduArticles
from tests.integration.corpus.conftest import first_articles, small_stream


@pytest.mark.datasets
def test_streams_pages_with_paragraphs() -> None:
    articles = first_articles(FineWebEduArticles(small_stream, seed=0))

    assert len(articles) == 2
    assert all(article.text and "\n\n\n" not in article.text for article in articles)
