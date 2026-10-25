import pytest

from contcheck.corpus.stackexchange import StackExchangeArticles
from tests.integration.corpus.conftest import first_articles, small_stream


@pytest.mark.datasets
def test_streams_highly_voted_answers() -> None:
    source = StackExchangeArticles(small_stream, seed=0, sites=("travel", "cooking"))

    articles = first_articles(source)

    assert len(articles) == 2
    assert all(article.text for article in articles)
