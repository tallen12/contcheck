import pytest

from contcheck.corrupt.edit import Edit
from contcheck.taxonomy.kinds import ErrorKind, Formality, Operation


def test_valid_edits_for_each_operation() -> None:
    Edit(2, 3, ErrorKind.VERB_TENSE, Operation.REPLACEMENT, ("went",), ("go",))
    Edit(1, 2, ErrorKind.DETERMINER, Operation.MISSING, ("the",), ())
    Edit(1, 1, ErrorKind.DETERMINER, Operation.UNNECESSARY, (), ("the",))


@pytest.mark.parametrize(
    ("start", "end", "op", "orig", "new"),
    [
        (2, 1, Operation.REPLACEMENT, (), ("x",)),
        (0, 2, Operation.REPLACEMENT, ("a",), ("b",)),
        (0, 1, Operation.REPLACEMENT, ("a",), ("a",)),
        (0, 1, Operation.REPLACEMENT, ("a",), ()),
        (0, 1, Operation.MISSING, ("a",), ("b",)),
        (0, 0, Operation.UNNECESSARY, (), ()),
    ],
)
def test_inconsistent_edits_are_rejected(
    start: int, end: int, op: Operation, orig: tuple[str, ...], new: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        Edit(start, end, ErrorKind.SPELLING, op, orig, new)


def test_edits_are_errors_unless_marked_informal() -> None:
    plain = Edit(1, 2, ErrorKind.ADJECTIVE_ADVERB, Operation.REPLACEMENT, ("quickly",), ("quick",))
    informal = Edit(
        1,
        2,
        ErrorKind.ADJECTIVE_ADVERB,
        Operation.REPLACEMENT,
        ("quickly",),
        ("quick",),
        Formality.INFORMAL,
    )

    assert plain.formality is Formality.ERROR
    assert informal.formality is Formality.INFORMAL
