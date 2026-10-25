"""Fakes used only by infer tests."""

from collections.abc import Sequence

from contcheck.infer.models import SpanError
from contcheck.taxonomy.kinds import ErrorKind, Formality


class FlatAdverbs:
    """Calls AdjectiveAdverb flags on quick or slow informal."""

    def classify(self, words: Sequence[str], error: SpanError) -> Formality:
        flagged = " ".join(words[error.start : error.end]).lower()
        if error.kind is ErrorKind.ADJECTIVE_ADVERB and flagged in {"quick", "slow"}:
            return Formality.INFORMAL
        return Formality.ERROR
