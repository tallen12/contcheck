"""Interfaces for masked language models."""

from collections.abc import Sequence
from typing import Protocol

from contcheck.lm.models import MaskedSlot, SlotPrediction


class MaskedScorer(Protocol):
    """Predicts and scores words at a masked position, given the words around it."""

    def predict(self, slots: Sequence[MaskedSlot], *, top_k: int) -> list[SlotPrediction]:
        """One prediction per slot, in order; adapters batch the slots.

        Each prediction has up to ``top_k`` most probable words (``top_k=0`` only scores
        the candidates) and the probability of each candidate.
        """
        ...
