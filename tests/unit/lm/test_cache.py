import subprocess
import sys
from collections.abc import Sequence

import pytest

from contcheck.lm.cache import CachedScorer
from contcheck.lm.models import MaskedSlot, SlotPrediction, WordScore


class CountingScorer:
    """Scores each candidate by its length and records every batch it is asked."""

    def __init__(self) -> None:
        self.batches: list[list[MaskedSlot]] = []

    def predict(self, slots: Sequence[MaskedSlot], *, top_k: int) -> list[SlotPrediction]:
        self.batches.append(list(slots))
        return [
            SlotPrediction(
                tuple(WordScore(f"w{k}", 0.1) for k in range(top_k)),
                tuple(WordScore(word, len(word) / 10) for word in slot.candidates),
            )
            for slot in slots
        ]


def slot(word: str) -> MaskedSlot:
    return MaskedSlot(("the", word), (True, False), 1, (word,))


def test_repeated_slots_are_scored_once() -> None:
    inner = CountingScorer()
    cache = CachedScorer(inner)

    first = cache.predict([slot("dog"), slot("cat"), slot("dog")], top_k=2)
    second = cache.predict([slot("cat")], top_k=2)

    assert [len(batch) for batch in inner.batches] == [2]
    assert first[0] == first[2]
    assert second[0] == first[1]


def test_top_k_is_part_of_the_key() -> None:
    inner = CountingScorer()
    cache = CachedScorer(inner)

    cache.predict([slot("dog")], top_k=1)
    wider = cache.predict([slot("dog")], top_k=3)

    assert len(inner.batches) == 2
    assert len(wider[0].top) == 3


def test_least_recent_entries_are_evicted() -> None:
    inner = CountingScorer()
    cache = CachedScorer(inner, max_entries=2)

    cache.predict([slot("a")], top_k=0)
    cache.predict([slot("b")], top_k=0)
    cache.predict([slot("a")], top_k=0)
    cache.predict([slot("c")], top_k=0)
    cache.predict([slot("a")], top_k=0)
    cache.predict([slot("b")], top_k=0)

    assert [batch[0].words[1] for batch in inner.batches] == ["a", "b", "c", "b"]


def test_cache_needs_room() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CachedScorer(CountingScorer(), max_entries=0)


def test_scoring_modules_do_not_import_torch() -> None:
    code = (
        "import sys, contcheck.lm.cache, contcheck.lm.models, contcheck.lm.protocols; "
        "sys.exit('torch' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
