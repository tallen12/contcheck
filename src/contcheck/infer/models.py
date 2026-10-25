"""The errors a checker reports."""

from dataclasses import dataclass

from contcheck.taxonomy.kinds import ErrorKind


@dataclass(frozen=True, slots=True)
class SpanError:
    """Words ``start`` to ``end`` (exclusive) look like an error of ``kind``.

    ``probability`` is the model's mean probability for the span's labels. The operation
    (Missing, Unnecessary, Replacement) is not predicted yet; the tagger only finds kinds.
    """

    start: int
    end: int
    kind: ErrorKind
    probability: float
