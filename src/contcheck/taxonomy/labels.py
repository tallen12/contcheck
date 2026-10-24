"""The word-level BIO label set derived from :class:`ErrorKind`.

There are ``1 + 2 * len(ErrorKind)`` labels: ``O`` plus ``B-<kind>`` and ``I-<kind>`` for
every kind. Label ids follow the order of :data:`LABELS` and are what the model predicts.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from contcheck.taxonomy.kinds import ErrorKind

OUTSIDE = "O"


class Prefix(StrEnum):
    """Position of a word inside an error span."""

    BEGIN = "B"
    INSIDE = "I"


@dataclass(frozen=True, slots=True)
class Outside:
    """A word outside every error span."""


@dataclass(frozen=True, slots=True)
class Tagged:
    """A word inside an error span of ``kind``."""

    prefix: Prefix
    kind: ErrorKind


type Label = Outside | Tagged


def format_label(label: Label) -> str:
    """Return the text form of ``label``, for example ``B-VerbTense``."""
    match label:
        case Outside():
            return OUTSIDE
        case Tagged(prefix=prefix, kind=kind):
            return f"{prefix.value}-{kind.value}"


def parse_label(text: str) -> Label:
    """Parse the text form of a label. Raises ``ValueError`` for unknown labels."""
    if text == OUTSIDE:
        return Outside()
    prefix, sep, kind = text.partition("-")
    if not sep:
        raise ValueError(f"unknown label {text!r}")
    try:
        return Tagged(Prefix(prefix), ErrorKind(kind))
    except ValueError as error:
        raise ValueError(f"unknown label {text!r}") from error


LABELS: tuple[str, ...] = (
    OUTSIDE,
    *(format_label(Tagged(prefix, kind)) for kind in ErrorKind for prefix in Prefix),
)
LABEL2ID: Mapping[str, int] = MappingProxyType({label: i for i, label in enumerate(LABELS)})
ID2LABEL: Mapping[int, str] = MappingProxyType(dict(enumerate(LABELS)))
