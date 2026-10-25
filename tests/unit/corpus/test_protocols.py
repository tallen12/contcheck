from contcheck.corpus.protocols import ArticleSource
from tests.unit.corpus.conftest import FakeArticleSource


def test_article_source_contract(articles: FakeArticleSource) -> None:
    source: ArticleSource = articles

    assert [a.id for a in source.articles()] == ["1", "2", "3"]
    assert list(source.articles()) == list(source.articles())
