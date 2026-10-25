import pytest

from contcheck.annotation.convert import shape_operation, to_annotation
from contcheck.annotation.models import AlignmentError, Annotated, TypedEdit, Unmappable
from contcheck.corpus.record import EditRecord
from contcheck.taxonomy.errant_map import EditEvidence
from contcheck.taxonomy.kinds import ErrorKind, Operation


def typed(
    noisy: tuple[int, int], clean: tuple[int, int], code: str, *, words: tuple[str, ...] = ()
) -> TypedEdit:
    tags = tuple("DT" for _ in words)
    return TypedEdit(
        noisy_start=noisy[0],
        noisy_end=noisy[1],
        clean_start=clean[0],
        clean_end=clean[1],
        code=code,
        evidence=EditEvidence(
            noisy=words,
            clean=words,
            noisy_tags=tags,
            clean_tags=tags,
            noisy_deps=tags,
            clean_deps=tags,
            noisy_is_word=True,
        ),
    )


NOISY = ("I", "go", "the", "store", "yesterday")
CLEAN = ("I", "went", "to", "the", "store", "yesterday")


def test_edits_become_records_in_noisy_indices() -> None:
    edits = [typed((2, 2), (2, 3), "M:PREP"), typed((1, 2), (1, 2), "R:VERB:TENSE")]

    result = to_annotation(edits, NOISY, CLEAN)

    assert result == Annotated(
        (
            EditRecord(
                start=1, end=2, kind=ErrorKind.VERB_TENSE, op=Operation.REPLACEMENT, orig=("went",)
            ),
            EditRecord(
                start=2, end=2, kind=ErrorKind.PREPOSITION, op=Operation.MISSING, orig=("to",)
            ),
        )
    )


def test_unnecessary_edits_have_no_clean_words() -> None:
    result = to_annotation([typed((1, 2), (1, 1), "U:DET")], ("a", "the", "dog"), ("a", "dog"))

    assert result == Annotated(
        (EditRecord(start=1, end=2, kind=ErrorKind.DETERMINER, op=Operation.UNNECESSARY, orig=()),)
    )


def test_identical_words_have_no_edits() -> None:
    assert to_annotation([], CLEAN, CLEAN) == Annotated(())


def test_a_dropped_edit_makes_the_pair_unmappable() -> None:
    edits = [typed((1, 2), (1, 2), "R:VERB:TENSE"), typed((2, 2), (2, 3), "UNK")]

    result = to_annotation(edits, NOISY, CLEAN)

    assert result == Unmappable("UNK", (), ("to",), "detected but not corrected")


def test_operation_follows_the_range_not_the_code() -> None:
    # ERRANT types [Doctor -> The doctor] as M:DET though it replaces a word.
    edit = typed((0, 1), (0, 2), "M:DET")

    result = to_annotation([edit], ("Doctor", "said"), ("The", "doctor", "said"))

    assert shape_operation(edit) is Operation.REPLACEMENT
    assert result == Annotated(
        (
            EditRecord(
                start=0,
                end=1,
                kind=ErrorKind.DETERMINER,
                op=Operation.REPLACEMENT,
                orig=("The", "doctor"),
            ),
        )
    )


def test_edits_that_do_not_restore_the_clean_words_fail_loudly() -> None:
    with pytest.raises(AlignmentError):
        to_annotation([typed((1, 2), (1, 2), "R:VERB:TENSE")], NOISY, CLEAN)
