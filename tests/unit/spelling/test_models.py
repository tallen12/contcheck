import dataclasses

import pytest

from contcheck.spelling.models import Suggestion


def test_suggestion_is_immutable() -> None:
    suggestion = Suggestion("the", 1, 100)

    with pytest.raises(dataclasses.FrozenInstanceError):
        suggestion.term = "they"  # ty: ignore[invalid-assignment] -- checking the runtime guard
