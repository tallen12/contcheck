from contcheck.corpus.cleanliness import ACCEPT_ALL
from contcheck.corpus.models import Clean
from contcheck.corpus.protocols import ArticleSource, ParagraphScreen
from tests.unit.corpus.conftest import FakeArticleSource


def test_article_source_contract(articles: FakeArticleSource) -> None:
    source: ArticleSource = articles

    assert [a.id for a in source.articles()] == ["1", "2", "3"]
    assert list(source.articles()) == list(source.articles())


def test_paragraph_screen_contract() -> None:
    screen: ParagraphScreen = ACCEPT_ALL

    assert screen.screen("Any text.") == Clean()
