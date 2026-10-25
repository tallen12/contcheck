"""Clean the corpus before corruption: apply the error changes ``corpus annotate`` found.

The pipeline is corpus -> corrected corpus -> ``corrupt``. Errors left in the clean text
become false negatives in training, so each paragraph gets the ``error`` changes the model
proposed. A rewrite is fine as long as the result is correct. Informal uses stay as
written, since informal edits are not labeled yet (see ``corrupt.apply``).

Some changes break correct text in ways a check can see without a model: doubled
punctuation, or a sentence start put in lower case. A paragraph with such a change is
dropped, because which of its changes are right is then unknown. Paragraphs whose
annotation failed are not in the annotated files, so they are left out too.
"""

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from pydantic import BaseModel, ConfigDict

from contcheck.corpus.annotate import apply_changes, input_files
from contcheck.corpus.annotated import AnnotatedParagraph, Change
from contcheck.corpus.documents import RawParagraph
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.corpus.readers import normalize_whitespace
from contcheck.taxonomy.kinds import Formality

DROPPED_DIR = "dropped"
DOUBLED_PUNCTUATION = re.compile(r"[.,;:!?] ?[.,;:]")
SENTENCE_END = re.compile("[.!?][\"'\u201d\u2019)\\]]* $")
"""A sentence end, perhaps after closing quotes or brackets, then a space."""


class DropReason(StrEnum):
    """Why a paragraph was left out of the corrected corpus."""

    DOUBLED_PUNCTUATION = "doubled_punctuation"
    """A change left two punctuation marks together ("string.." or "2012,,")."""
    LOWERCASED_SENTENCE_START = "lowercased_sentence_start"
    """A change put the first word of a sentence in lower case."""


@dataclass(frozen=True, slots=True)
class Unchanged:
    """The model found no error (perhaps informal uses, left as written)."""

    paragraph: RawParagraph


@dataclass(frozen=True, slots=True)
class Corrected:
    """The paragraph with its error changes applied; ``changes`` counts them."""

    paragraph: RawParagraph
    changes: int


class DroppedParagraph(BaseModel):
    """One line of the drop log: a paragraph left out, with its proposed correction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    hash: str
    reason: DropReason
    detail: str
    text: str
    corrected: str


@dataclass(frozen=True, slots=True)
class Dropped:
    """A change broke the text, so the paragraph is left out."""

    dropped: DroppedParagraph


type Correction = Unchanged | Corrected | Dropped


def error_changes(paragraph: AnnotatedParagraph) -> tuple[Change, ...]:
    """The changes marked ``error``; informal ones are not applied."""
    return tuple(change for change in paragraph.changes if change.type is Formality.ERROR)


def corrected_text(paragraph: AnnotatedParagraph) -> str:
    """The paragraph with its error changes applied."""
    return normalize_whitespace(apply_changes(paragraph.text, error_changes(paragraph)))


@dataclass(frozen=True, slots=True)
class Breakage:
    """A change that breaks correct text, found without a model."""

    reason: DropReason
    detail: str


def breakage(text: str, changes: Sequence[Change], corrected: str) -> Breakage | None:
    """Why ``changes`` break ``text``, or ``None`` when no check objects."""
    doubled = len(DOUBLED_PUNCTUATION.findall(corrected)) - len(DOUBLED_PUNCTUATION.findall(text))
    if doubled > 0:
        return Breakage(DropReason.DOUBLED_PUNCTUATION, f"{doubled} new doubled marks")
    for change in changes:
        if _lowercases_sentence_start(text, change):
            detail = f"{change.original!r} -> {change.replacement!r}"
            return Breakage(DropReason.LOWERCASED_SENTENCE_START, detail)
    return None


def _lowercases_sentence_start(text: str, change: Change) -> bool:
    # Only the first letter lowered ("Specifically" -> "specifically"); "Ph" -> "pH" is a fix.
    word = change.original.split(" ", 1)[0]
    if not word[:1].isupper() or not change.replacement.startswith(word[:1].lower() + word[1:]):
        return False
    before = text[: change.start]
    return not before or SENTENCE_END.search(before) is not None


def correct_paragraph(paragraph: AnnotatedParagraph) -> Correction:
    """Apply one paragraph's error changes, or drop it when a change breaks the text."""
    changes = error_changes(paragraph)
    corrected = corrected_text(paragraph)
    if corrected == paragraph.text:
        return Unchanged(RawParagraph(source_id=paragraph.source_id, text=paragraph.text))
    broken = breakage(paragraph.text, changes, corrected)
    if broken is not None:
        return Dropped(
            DroppedParagraph(
                source_id=paragraph.source_id,
                hash=paragraph.hash,
                reason=broken.reason,
                detail=broken.detail,
                text=paragraph.text,
                corrected=corrected,
            )
        )
    return Corrected(RawParagraph(source_id=paragraph.source_id, text=corrected), len(changes))


def group_by_source(paragraphs: Sequence[AnnotatedParagraph]) -> list[AnnotatedParagraph]:
    """Paragraphs with each source's paragraphs together, in order of first appearance.

    A paragraph whose annotation failed and was retried in a later run is appended after
    the rest; grouping keeps it with its source, as ``group_documents`` requires.
    """
    by_source: dict[str, list[AnnotatedParagraph]] = {}
    for paragraph in paragraphs:
        by_source.setdefault(paragraph.source_id, []).append(paragraph)
    return [paragraph for group in by_source.values() for paragraph in group]


@dataclass(slots=True)
class CorrectionTally:
    """Counts over every input file."""

    unchanged: int = 0
    corrected: int = 0
    changes: int = 0
    dropped: Counter[DropReason] = field(default_factory=Counter[DropReason])

    def add(self, correction: Correction) -> None:
        match correction:
            case Unchanged():
                self.unchanged += 1
            case Corrected(changes=changes):
                self.corrected += 1
                self.changes += changes
            case Dropped(dropped):
                self.dropped[dropped.reason] += 1
            case _:
                assert_never(correction)

    def summary(self) -> str:
        """Paragraph counts, then drops by reason."""
        total = self.unchanged + self.corrected + sum(self.dropped.values())
        drops = ", ".join(f"{reason.value} {n}" for reason, n in self.dropped.most_common())
        return (
            f"{total} paragraphs: {self.unchanged} unchanged,"
            f" {self.corrected} corrected ({self.changes} changes),"
            f" {sum(self.dropped.values())} dropped ({drops or 'none'})"
        )


def correct_files(source: Path, out_dir: Path) -> CorrectionTally:
    """Correct every annotated file in ``source`` (a JSONL file or a folder of them).

    Each input file's kept paragraphs go to ``out_dir/<name>`` as raw paragraphs, ready
    for ``corrupt --in out_dir``; dropped ones go to ``out_dir/dropped/<name>``.
    """
    tally = CorrectionTally()
    for file in input_files(source):
        paragraphs = group_by_source(list(read_jsonl(file, AnnotatedParagraph)))
        corrections = [correct_paragraph(paragraph) for paragraph in paragraphs]
        for correction in corrections:
            tally.add(correction)
        write_jsonl(out_dir / file.name, _kept(corrections))
        write_jsonl(out_dir / DROPPED_DIR / file.name, _dropped_lines(corrections))
    return tally


def _kept(corrections: Iterable[Correction]) -> Iterator[RawParagraph]:
    for correction in corrections:
        match correction:
            case Unchanged(paragraph) | Corrected(paragraph=paragraph):
                yield paragraph
            case Dropped():
                pass
            case _:
                assert_never(correction)


def _dropped_lines(corrections: Iterable[Correction]) -> Iterator[DroppedParagraph]:
    for correction in corrections:
        if isinstance(correction, Dropped):
            yield correction.dropped
