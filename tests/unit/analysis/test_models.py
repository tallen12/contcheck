import pytest

from contcheck.analysis.models import AnalyzedText, Token, join_words


def token(text: str, *, space_after: bool) -> Token:
    return Token(text, "NN", text, space_after=space_after, sentence_start=False, finite=False)


def test_analyzed_text_rebuilds_words_and_text() -> None:
    analyzed = AnalyzedText(
        (
            token("Hi", space_after=False),
            token(",", space_after=True),
            token("you", space_after=False),
        )
    )

    assert analyzed.words == ("Hi", ",", "you")
    assert analyzed.text == "Hi, you"


def test_join_words_needs_one_flag_per_word() -> None:
    with pytest.raises(ValueError, match="space flags"):
        join_words(("a", "b"), (True,))
