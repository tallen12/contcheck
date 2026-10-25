"""Interfaces for refining decoded errors."""

from collections.abc import Sequence
from typing import Protocol

from contcheck.infer.models import SpanError
from contcheck.taxonomy.kinds import Formality


class FormalityClassifier(Protocol):
    """Decides whether a flagged span is an error in any register or only informal."""

    def classify(self, words: Sequence[str], error: SpanError) -> Formality:
        """The formality of ``error`` among ``words``, the text it was found in."""
        ...
