import pytest

from contcheck.corpus.cleanup import has_dropped_text, join_paragraphs, lines_to_paragraphs


def test_join_paragraphs_uses_blank_lines() -> None:
    assert join_paragraphs(["One.", "Two."]) == "One.\n\nTwo."


def test_every_non_blank_line_becomes_a_paragraph() -> None:
    text = "Understand\nThe city lies on a river.\n\n  Get in  \n"

    assert lines_to_paragraphs(text) == "Understand\n\nThe city lies on a river.\n\nGet in"


@pytest.mark.parametrize(
    "line",
    [
        "the  of the People's Republic of China",
        "with the president of Brazil,, at the residence",
        "Some ministers were present (Economy) , and others.",
        "the company ( president) spoke.",
    ],
)
def test_gaps_from_dropped_links_are_detected(line: str) -> None:
    assert has_dropped_text(line)


def test_ordinary_prose_has_no_gaps() -> None:
    assert not has_dropped_text('He said, "Lunch was (mostly) meat." Then he left.')
