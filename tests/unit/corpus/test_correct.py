from pathlib import Path

import pytest

from contcheck.corpus.annotate import apply_changes
from contcheck.corpus.annotated import AnnotatedParagraph, Change, paragraph_hash
from contcheck.corpus.correct import (
    DROPPED_DIR,
    Corrected,
    CorrectionTally,
    Dropped,
    DroppedParagraph,
    DropReason,
    Unchanged,
    correct_files,
    correct_paragraph,
    corrected_text,
    error_changes,
    group_by_source,
)
from contcheck.corpus.documents import RawParagraph
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.corpus.readers import read_paragraphs
from contcheck.taxonomy.kinds import Formality


def annotated(
    text: str, *changes: tuple[str, str, Formality], source_id: str = "src-1"
) -> AnnotatedParagraph:
    """An annotation of ``text`` with each ``(original, replacement, type)`` located."""
    located: list[Change] = []
    for original, replacement, kind in changes:
        start = text.index(original)
        located.append(
            Change(
                start=start,
                end=start + len(original),
                original=original,
                replacement=replacement,
                type=kind,
            )
        )
    return AnnotatedParagraph(
        source_id=source_id,
        hash=paragraph_hash(text),
        text=text,
        changes=tuple(located),
        corrected=apply_changes(text, located),
        model="fake",
        attempts=1,
    )


def test_only_error_changes_are_applied() -> None:
    paragraph = annotated(
        "Their going home and gonna rest.",
        ("Their", "They're", Formality.ERROR),
        ("gonna", "going to", Formality.INFORMAL),
    )

    assert [c.original for c in error_changes(paragraph)] == ["Their"]
    assert corrected_text(paragraph) == "They're going home and gonna rest."


def test_paragraph_without_errors_is_unchanged() -> None:
    informal = annotated("We gonna go.", ("gonna", "are going to", Formality.INFORMAL))

    assert correct_paragraph(annotated("The dog went home.")) == Unchanged(
        RawParagraph(source_id="src-1", text="The dog went home.")
    )
    assert correct_paragraph(informal) == Unchanged(
        RawParagraph(source_id="src-1", text="We gonna go.")
    )


@pytest.mark.parametrize(
    ("text", "change", "corrected"),
    [
        ("It is big then a house.", ("then", "than"), "It is big than a house."),
        (
            "Tornadoes have been reported to have taken place in Ohio.",
            ("taken place", "occurred"),
            "Tornadoes have been reported to have occurred in Ohio.",
        ),
        (
            "Now, open a browser web and log in.",
            ("browser web", "browser"),
            "Now, open a browser and log in.",
        ),
        ("Ph is an indication of acidity.", ("Ph", "pH"), "pH is an indication of acidity."),
    ],
)
def test_error_changes_and_rewrites_are_applied(
    text: str, change: tuple[str, str], corrected: str
) -> None:
    paragraph = annotated(text, (*change, Formality.ERROR))

    assert correct_paragraph(paragraph) == Corrected(
        RawParagraph(source_id="src-1", text=corrected), 1
    )


@pytest.mark.parametrize(
    ("text", "change", "reason"),
    [
        (
            "It marks the end of the string. The type follows.",
            ("end of the string", "end of the string."),
            DropReason.DOUBLED_PUNCTUATION,
        ),
        (
            "By 2012, the value will rise.",
            ("By 2012", "By 2012,"),
            DropReason.DOUBLED_PUNCTUATION,
        ),
        (
            'It uses CC-BY 2.5. Specifically "CC-BY-3.0" is fine.',
            ("Specifically", "specifically"),
            DropReason.LOWERCASED_SENTENCE_START,
        ),
        (
            "Northwestern parts are cold.",
            ("Northwestern", "northwestern"),
            DropReason.LOWERCASED_SENTENCE_START,
        ),
    ],
)
def test_changes_that_break_the_text_drop_the_paragraph(
    text: str, change: tuple[str, str], reason: DropReason
) -> None:
    paragraph = annotated(text, (*change, Formality.ERROR))

    result = correct_paragraph(paragraph)

    assert isinstance(result, Dropped)
    assert result.dropped.reason is reason
    assert result.dropped.text == text
    assert result.dropped.hash == paragraph.hash
    assert result.dropped.corrected == corrected_text(paragraph)


def test_lowercasing_inside_a_sentence_is_kept() -> None:
    paragraph = annotated(
        "Gould is in the Northwestern part.", ("Northwestern", "northwestern", Formality.ERROR)
    )

    assert isinstance(correct_paragraph(paragraph), Corrected)


def test_existing_doubled_punctuation_does_not_drop() -> None:
    paragraph = annotated("Wait.. it is then fine.", ("then", "than", Formality.ERROR))

    assert isinstance(correct_paragraph(paragraph), Corrected)


def test_paragraphs_are_grouped_by_source_in_first_seen_order() -> None:
    first, second, retried = (
        annotated("One.", source_id="a"),
        annotated("Two.", source_id="b"),
        annotated("Three.", source_id="a"),
    )

    assert group_by_source([first, second, retried]) == [first, retried, second]


def test_tally_summarizes_outcomes() -> None:
    tally = CorrectionTally()
    for paragraph in [
        annotated("Fine."),
        annotated("It is big then a house.", ("then", "than", Formality.ERROR)),
        annotated("Ok. So it goes.", ("So", "so", Formality.ERROR)),
    ]:
        tally.add(correct_paragraph(paragraph))

    assert tally.summary() == (
        "3 paragraphs: 1 unchanged, 1 corrected (1 changes),"
        " 1 dropped (lowercased_sentence_start 1)"
    )


def test_correct_files_writes_the_corpus_and_a_drop_log(tmp_path: Path) -> None:
    source = tmp_path / "annotated"
    write_jsonl(
        source / "web.jsonl",
        [
            annotated("It is big then a house.", ("then", "than", Formality.ERROR), source_id="w"),
            annotated("Ok. So it goes.", ("So", "so", Formality.ERROR), source_id="v"),
            annotated("Fine.", source_id="w"),
        ],
    )
    write_jsonl(source / "failed" / "web.jsonl", [])
    out = tmp_path / "corrected"

    tally = correct_files(source, out)

    assert (tally.unchanged, tally.corrected) == (1, 1)
    assert list(read_paragraphs(out)) == [
        RawParagraph(source_id="w", text="It is big than a house."),
        RawParagraph(source_id="w", text="Fine."),
    ]
    [dropped] = read_jsonl(out / DROPPED_DIR / "web.jsonl", DroppedParagraph)
    assert dropped.text == "Ok. So it goes."
