"""A ``MaskedScorer`` that remembers recent predictions."""

from collections import OrderedDict
from collections.abc import Sequence

from contcheck.lm.models import MaskedSlot, SlotPrediction
from contcheck.lm.protocols import MaskedScorer

DEFAULT_ENTRIES = 4096


class CachedScorer:
    """Wraps a scorer and reuses predictions for slots it has seen, least recent out first.

    Generation asks about the same paragraph for each variant and on every retry, so most
    repeat questions come close together and a small cache catches them.
    """

    def __init__(self, scorer: MaskedScorer, *, max_entries: int = DEFAULT_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("the cache needs room for at least one entry")
        self._scorer = scorer
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[MaskedSlot, int], SlotPrediction] = OrderedDict()

    def predict(self, slots: Sequence[MaskedSlot], *, top_k: int) -> list[SlotPrediction]:
        keys = [(slot, top_k) for slot in slots]
        missing = list(dict.fromkeys(key for key in keys if key not in self._entries))
        if missing:
            fresh = self._scorer.predict([slot for slot, _ in missing], top_k=top_k)
            for key, prediction in zip(missing, fresh, strict=True):
                self._entries[key] = prediction
        predictions = []
        for key in keys:
            self._entries.move_to_end(key)
            predictions.append(self._entries[key])
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return predictions
