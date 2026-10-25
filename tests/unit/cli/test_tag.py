from contcheck.cli.tag import format_flags, split_paragraphs
from contcheck.infer.models import SpanError
from contcheck.taxonomy.kinds import ErrorKind


def test_format_flags_lines_up_spans() -> None:
    words = ["Yesterday", "I", "go", "to", "the", "store", "and", "buyed", "apples"]
    errors = [
        SpanError(2, 3, ErrorKind.VERB_TENSE, 0.93),
        SpanError(7, 8, ErrorKind.SPELLING, 0.991),
    ]

    assert format_flags(words, errors) == (
        "go     VerbTense              0.93\nbuyed  Spelling               0.99"
    )


def test_format_flags_without_errors() -> None:
    assert format_flags(["Fine", "."], []) == "no errors flagged"


def test_split_paragraphs_on_blank_lines() -> None:
    text = "First one\nwraps here.\n\n  \nSecond one.\n"

    assert split_paragraphs(text) == ["First one wraps here.", "Second one."]
    assert split_paragraphs(" \n\n ") == []
