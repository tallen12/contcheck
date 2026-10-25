"""One kind of error drawn from several corruptors, mixed by weight."""

from collections.abc import Sequence

from contcheck.corrupt.apply import weighted_choice
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Corruptor, Rng
from contcheck.taxonomy.kinds import ErrorKind


class CompositeCorruptor:
    """Asks one of its corruptors, chosen by weight, for an edit of their shared kind.

    When the chosen corruptor finds no site, the others are tried in turn, again by
    weight, so a paragraph only loses the kind when none of them can corrupt it. The
    realized mix therefore leans toward corruptors that find sites more often.
    """

    def __init__(self, kind: ErrorKind, parts: Sequence[tuple[Corruptor, float]]) -> None:
        if any(corruptor.kind is not kind for corruptor, _ in parts):
            raise ValueError(f"every part must produce {kind}")
        if any(weight < 0 for _, weight in parts) or sum(w for _, w in parts) <= 0:
            raise ValueError("weights must be non-negative with a positive total")
        self._kind = kind
        self._parts = [(corruptor, weight) for corruptor, weight in parts if weight > 0]

    @property
    def kind(self) -> ErrorKind:
        return self._kind

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        remaining = {index: weight for index, (_, weight) in enumerate(self._parts)}
        while remaining:
            index = weighted_choice(remaining, rng)
            edit = self._parts[index][0].corrupt(context, rng)
            if edit is not None:
                return edit
            del remaining[index]
        return None
