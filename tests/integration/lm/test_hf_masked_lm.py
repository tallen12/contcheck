import pytest

from contcheck.cli.device import choose_device, visible_gpus
from contcheck.lm.hf_masked_lm import load_masked_scorer
from contcheck.lm.models import MaskedSlot


@pytest.mark.models
def test_distilroberta_fills_an_obvious_gap() -> None:
    scorer = load_masked_scorer(device=choose_device(None, visible_gpus()))
    words = ("Paris", "is", "the", "capital", "of", "France", ".")
    spaces = (True, True, True, True, True, False, False)

    start, middle = scorer.predict(
        [
            MaskedSlot(words, spaces, 0, ("Paris", "London")),
            MaskedSlot(words, spaces, 3, ("capital", "banana")),
        ],
        top_k=10,
    )

    assert "capital" in [score.word for score in middle.top]
    assert (middle.probability("capital") or 0) > 100 * (middle.probability("banana") or 1)
    assert (start.probability("Paris") or 0) > (start.probability("London") or 1)
