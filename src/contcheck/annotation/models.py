"""Value types produced by annotation."""

from dataclasses import dataclass

from contcheck.corpus.record import EditRecord
from contcheck.taxonomy.errant_map import EditEvidence


@dataclass(frozen=True, slots=True)
class TypedEdit:
    """One edit as ERRANT reports it: word ranges on both sides and its code.

    ``noisy_start:noisy_end`` and ``clean_start:clean_end`` index the noisy and clean words;
    one range is empty for Missing and Unnecessary edits.
    """

    noisy_start: int
    noisy_end: int
    clean_start: int
    clean_end: int
    code: str
    evidence: EditEvidence


@dataclass(frozen=True, slots=True)
class Annotated:
    """Every edit mapped onto our taxonomy, in noisy word indices, ordered by position."""

    edits: tuple[EditRecord, ...]


@dataclass(frozen=True, slots=True)
class Unmappable:
    """An edit ERRANT found has no kind of ours, so the pair cannot be labeled."""

    code: str
    noisy: tuple[str, ...]
    clean: tuple[str, ...]
    reason: str


type Annotation = Annotated | Unmappable


class AlignmentError(RuntimeError):
    """The mapped edits do not turn the noisy words into the clean words.

    ERRANT works on exactly the words it is given, so this means a bug in the adapter or
    an ERRANT change, never bad input; it is raised rather than returned.
    """
