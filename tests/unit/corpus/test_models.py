import pytest
from pydantic import ValidationError

from contcheck.corpus.models import Article, Clean, Rejected, RejectReason


def test_article_ignores_extra_columns() -> None:
    article = Article.model_validate({"id": "12", "title": "A", "text": "B", "url": "u"})

    assert article == Article(id="12", title="A", text="B")


def test_article_needs_text() -> None:
    with pytest.raises(ValidationError):
        Article.model_validate({"id": "12", "title": "A"})


def test_verdicts_compare_by_value() -> None:
    assert Clean() == Clean()
    assert Rejected(RejectReason.LIST) != Rejected(RejectReason.MARKUP)
    assert RejectReason("non_words") is RejectReason.NON_WORDS
