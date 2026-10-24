from dataclasses import dataclass

import pytest

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.registry import CorruptorRegistry
from contcheck.taxonomy.kinds import ErrorKind


@dataclass(frozen=True)
class NoOp:
    kind: ErrorKind

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        return None


def test_registry_returns_kinds_in_taxonomy_order() -> None:
    registry = CorruptorRegistry([NoOp(ErrorKind.DETERMINER), NoOp(ErrorKind.SPELLING)])

    assert registry.kinds() == (ErrorKind.SPELLING, ErrorKind.DETERMINER)
    assert registry.get(ErrorKind.SPELLING).kind is ErrorKind.SPELLING
    assert ErrorKind.DETERMINER in registry
    assert ErrorKind.REAL_WORD not in registry


def test_registry_rejects_two_corruptors_for_one_kind() -> None:
    with pytest.raises(ValueError, match="two corruptors"):
        CorruptorRegistry([NoOp(ErrorKind.SPELLING), NoOp(ErrorKind.SPELLING)])


def test_unknown_kind_raises() -> None:
    with pytest.raises(KeyError):
        CorruptorRegistry([]).get(ErrorKind.SPELLING)
