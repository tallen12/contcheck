import random

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Corruptor, Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer


class DropFirstWord:
    kind = ErrorKind.MISSING_WORD

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        if not context.words:
            return None
        return Edit(0, 1, self.kind, Operation.MISSING, context.words[:1], ())


def test_corruptor_contract(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    corruptor: Corruptor = DropFirstWord()
    tense = TenseHistogram(0.0, 0.0, 0.0, 0.0)

    edit = corruptor.corrupt(CorruptionContext(analyzer.analyze("a dog").tokens, tense), rng)
    nothing = corruptor.corrupt(CorruptionContext((), tense), rng)

    assert edit is not None
    assert edit.kind is corruptor.kind
    assert nothing is None


def test_random_is_an_rng(rng: random.Random) -> None:
    source: Rng = rng

    assert 0 <= source.random() < 1
    assert 0 <= source.randrange(3) < 3
    assert source.choice(["x"]) == "x"
