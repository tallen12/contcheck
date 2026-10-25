from contcheck.corpus.fetch import ParagraphFilter, article_paragraphs, fetch_paragraphs
from contcheck.corpus.models import Article
from tests.unit.corpus.conftest import PROSE, FakeArticleSource


def test_filter_keeps_prose_only() -> None:
    paragraph_filter = ParagraphFilter(min_words=3, max_words=6)

    assert paragraph_filter.accepts("This is a sentence.")
    assert not paragraph_filter.accepts("Too short.")
    assert not paragraph_filter.accepts("No final punctuation here")
    assert not paragraph_filter.accepts("This one has far too many words in it.")


def test_article_paragraphs_drop_headings_and_lists() -> None:
    article = Article(id="1", title="t", text=f"Heading\n\n{PROSE}\n\n- item\n\n{PROSE}")

    assert article_paragraphs(article, ParagraphFilter()) == [PROSE, PROSE]


def test_fetch_skips_single_paragraph_articles_and_stops_at_limit(
    articles: FakeArticleSource,
) -> None:
    fetched = list(
        fetch_paragraphs(articles, prefix="wiki", limit=3, paragraph_filter=ParagraphFilter())
    )

    assert [p.source_id for p in fetched] == ["wiki-2", "wiki-2", "wiki-3"]


def test_fetch_with_no_limit_yields_nothing(articles: FakeArticleSource) -> None:
    assert (
        list(fetch_paragraphs(articles, prefix="w", limit=0, paragraph_filter=ParagraphFilter()))
        == []
    )


def test_fetch_caps_paragraphs_per_article(articles: FakeArticleSource) -> None:
    fetched = fetch_paragraphs(
        articles, prefix="w", limit=10, paragraph_filter=ParagraphFilter(), per_article=2
    )

    assert [p.source_id for p in fetched] == ["w-2", "w-2", "w-3", "w-3"]
