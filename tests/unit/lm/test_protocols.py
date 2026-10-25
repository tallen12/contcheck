from collections.abc import Sequence

from contcheck.lm.models import MaskedSlot, SlotPrediction, WordScore
from contcheck.lm.protocols import MaskedScorer


class UniformScorer:
    """Gives every candidate the same probability and proposes nothing."""

    def predict(self, slots: Sequence[MaskedSlot], *, top_k: int) -> list[SlotPrediction]:
        return [
            SlotPrediction((), tuple(WordScore(word, 0.1) for word in slot.candidates))
            for slot in slots
        ]


def test_scorer_contract() -> None:
    scorer: MaskedScorer = UniformScorer()
    slots = [
        MaskedSlot(("a", "dog"), (True, False), 1, ("dog", "cat")),
        MaskedSlot(("x",), (False,), 0),
    ]

    predictions = scorer.predict(slots, top_k=0)

    assert len(predictions) == 2
    assert predictions[0].probability("cat") == 0.1
    assert predictions[1].scored == ()
