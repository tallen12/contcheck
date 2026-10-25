import pytest

from contcheck.corpus.wikinews import WikinewsArticles
from tests.integration.corpus.conftest import first_articles, small_stream


@pytest.mark.datasets
def test_streams_english_articles() -> None:
    articles = first_articles(WikinewsArticles(small_stream, seed=0))

    assert len(articles) == 2
    assert all(article.text for article in articles)
