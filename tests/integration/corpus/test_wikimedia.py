import pytest

from contcheck.corpus.wikimedia import Wiki, WikimediaArticles
from tests.integration.corpus.conftest import first_articles, small_stream


@pytest.mark.datasets
@pytest.mark.parametrize("wiki", list(Wiki))
def test_streams_main_namespace_pages(wiki: Wiki) -> None:
    articles = first_articles(WikimediaArticles(small_stream, seed=0, wiki=wiki))

    assert len(articles) == 2
