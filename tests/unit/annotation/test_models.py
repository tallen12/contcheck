import dataclasses

import pytest

from contcheck.annotation.models import AlignmentError, Annotated, Unmappable


def test_results_are_immutable() -> None:
    result = Unmappable("UNK", ("a",), ("b",), "detected but not corrected")

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.code = "R:OTHER"  # ty: ignore[invalid-assignment] -- checking frozen at runtime
    assert Annotated(()) == Annotated(())


def test_alignment_errors_are_runtime_errors() -> None:
    assert issubclass(AlignmentError, RuntimeError)
