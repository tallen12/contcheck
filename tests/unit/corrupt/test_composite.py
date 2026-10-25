import random

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.composite import CompositeCorruptor
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)


class ReplaceWith:
    """Replaces the first word by ``new``, or finds no site when ``new`` is empty."""

    def __init__(self, new: str, kind: ErrorKind = ErrorKind.REAL_WORD) -> None:
        self._new = new
        self.kind = kind
        self.calls = 0

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        self.calls += 1
        if not self._new:
            return None
        return Edit(0, 1, self.kind, Operation.REPLACEMENT, context.words[:1], (self._new,))


@pytest.fixture
def context(analyzer: FakeAnalyzer) -> CorruptionContext:
    return CorruptionContext(analyzer.analyze("Then I walk.").tokens, NO_TENSE)


def test_parts_are_mixed_by_weight(context: CorruptionContext) -> None:
    composite = CompositeCorruptor(
        ErrorKind.REAL_WORD, [(ReplaceWith("than"), 3.0), (ReplaceWith("so"), 1.0)]
    )

    edits = [composite.corrupt(context, random.Random(seed)) for seed in range(400)]
    share = sum(e is not None and e.new == ("than",) for e in edits) / len(edits)

    assert composite.kind is ErrorKind.REAL_WORD
    assert 0.68 < share < 0.82


def test_falls_back_when_the_chosen_part_finds_no_site(
    context: CorruptionContext, rng: random.Random
) -> None:
    empty = ReplaceWith("")
    composite = CompositeCorruptor(ErrorKind.REAL_WORD, [(empty, 100.0), (ReplaceWith("so"), 1.0)])

    edits = [composite.corrupt(context, rng) for _ in range(20)]

    assert all(e is not None and e.new == ("so",) for e in edits)
    assert empty.calls > 0


def test_none_when_no_part_has_a_site(context: CorruptionContext, rng: random.Random) -> None:
    composite = CompositeCorruptor(ErrorKind.REAL_WORD, [(ReplaceWith(""), 1.0)])

    assert composite.corrupt(context, rng) is None


def test_zero_weight_parts_are_never_asked(context: CorruptionContext, rng: random.Random) -> None:
    unused = ReplaceWith("x")
    composite = CompositeCorruptor(ErrorKind.REAL_WORD, [(unused, 0.0), (ReplaceWith(""), 1.0)])

    assert composite.corrupt(context, rng) is None
    assert unused.calls == 0


def test_parts_must_share_the_kind_and_have_weight() -> None:
    with pytest.raises(ValueError, match="every part"):
        CompositeCorruptor(ErrorKind.REAL_WORD, [(ReplaceWith("x", ErrorKind.SPELLING), 1.0)])
    with pytest.raises(ValueError, match="positive total"):
        CompositeCorruptor(ErrorKind.REAL_WORD, [(ReplaceWith("x"), 0.0)])
    with pytest.raises(ValueError, match="non-negative"):
        CompositeCorruptor(ErrorKind.REAL_WORD, [(ReplaceWith("x"), -1.0), (ReplaceWith("y"), 2.0)])
