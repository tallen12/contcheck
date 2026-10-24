"""Corruptors indexed by the kind of error they produce."""

from collections.abc import Iterable

from contcheck.corrupt.protocols import Corruptor
from contcheck.taxonomy.kinds import ErrorKind


class CorruptorRegistry:
    """At most one corruptor per kind, built at the composition root."""

    def __init__(self, corruptors: Iterable[Corruptor]) -> None:
        self._by_kind: dict[ErrorKind, Corruptor] = {}
        for corruptor in corruptors:
            if corruptor.kind in self._by_kind:
                raise ValueError(f"two corruptors for {corruptor.kind}")
            self._by_kind[corruptor.kind] = corruptor

    def get(self, kind: ErrorKind) -> Corruptor:
        """The corruptor for ``kind``. Raises ``KeyError`` if none is registered."""
        return self._by_kind[kind]

    def kinds(self) -> tuple[ErrorKind, ...]:
        """Registered kinds in taxonomy order, so sampling does not depend on wiring."""
        return tuple(kind for kind in ErrorKind if kind in self._by_kind)

    def __contains__(self, kind: object) -> bool:
        return kind in self._by_kind
