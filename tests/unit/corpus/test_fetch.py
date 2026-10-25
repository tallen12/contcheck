from contcheck.corpus.fetch import (
    ParagraphFilter,
    article_paragraphs,
    fetch_paragraphs,
    screened_paragraphs,
)
from contcheck.corpus.models import Article, Clean, Rejected, RejectReason, Verdict
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


def test_fetch_skips_articles_whose_id_was_used() -> None:
    article = Article(id="7", title="", text=f"{PROSE}\n\n{PROSE}")
    source = FakeArticleSource([article, article])

    fetched = fetch_paragraphs(source, prefix="w", limit=10, paragraph_filter=ParagraphFilter())

    assert [p.source_id for p in fetched] == ["w-7", "w-7"]


class RejectShort:
    """A screen that rejects paragraphs starting with ``Short``."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def screen(self, paragraph: str) -> Verdict:
        self.seen.append(paragraph)
        if paragraph.startswith("Short"):
            return Rejected(RejectReason.NON_WORDS)
        return Clean()


def test_screen_drops_paragraphs_and_then_articles_left_too_short() -> None:
    short = PROSE.replace("This", "Short", 1)
    source = FakeArticleSource(
        [
            Article(id="1", title="t", text=f"{PROSE}\n\n{short}"),
            Article(id="2", title="t", text=f"{short}\n\n{PROSE}\n\n{PROSE}"),
        ]
    )

    fetched = fetch_paragraphs(
        source, prefix="w", limit=10, paragraph_filter=ParagraphFilter(), screen=RejectShort()
    )

    assert [(p.source_id, p.text) for p in fetched] == [("w-2", PROSE), ("w-2", PROSE)]


def test_screening_stops_once_an_article_has_enough_paragraphs() -> None:
    article = Article(id="1", title="t", text="\n\n".join([PROSE] * 5))
    screen = RejectShort()

    kept = screened_paragraphs(article, ParagraphFilter(), screen, limit=2)

    assert kept == [PROSE, PROSE]
    assert len(screen.seen) == 2
