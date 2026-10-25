import pytest

from contcheck.lm.models import MaskedSlot, SlotPrediction, WordScore


def test_slot_checks_spaces_and_position() -> None:
    with pytest.raises(ValueError, match="space flags"):
        MaskedSlot(("a", "dog"), (True,), 0)
    with pytest.raises(ValueError, match="outside"):
        MaskedSlot(("a", "dog"), (True, False), 2)


def test_prediction_looks_up_scored_words() -> None:
    prediction = SlotPrediction(
        top=(WordScore("cat", 0.5),), scored=(WordScore("dog", 0.25), WordScore("cow", 0.0))
    )

    assert prediction.probability("dog") == 0.25
    assert prediction.probability("cow") == 0.0
    assert prediction.probability("cat") is None
