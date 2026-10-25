import dataclasses

import pytest

from contcheck.infer.models import SpanError
from contcheck.taxonomy.kinds import ErrorKind


def test_span_error_is_an_immutable_value() -> None:
    error = SpanError(2, 3, ErrorKind.VERB_TENSE, 0.9)

    assert error == SpanError(2, 3, ErrorKind.VERB_TENSE, 0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        error.start = 1  # ty: ignore[invalid-assignment] -- checking the runtime guard
