"""Turning corruptor edits into noisy words and exact BIO labels.

All labeling happens here, so corruptors only decide what to change. Edits are kept
apart by at least one untouched word, which keeps every label span unambiguous.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contcheck.corpus.record import EditRecord
from contcheck.corrupt.config import GenConfig
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.registry import CorruptorRegistry
from contcheck.taxonomy.bio import WordSpan, missing_anchor, spans_to_labels
from contcheck.taxonomy.kinds import ErrorKind, Operation

ATTEMPTS_PER_ERROR = 4
"""How often to ask for another edit when one clashes with those already chosen."""


@dataclass(frozen=True, slots=True)
class Corrupted:
    """Noisy words with their trailing spaces, labels, and edits in noisy indices."""

    words: tuple[str, ...]
    spaces: tuple[bool, ...]
    labels: tuple[str, ...]
    edits: tuple[EditRecord, ...]


def weighted_choice[T](weights: Mapping[T, float], rng: Rng) -> T:
    """Pick a key with probability proportional to its weight. Weights must be positive."""
    total = sum(weights.values())
    if not weights or total <= 0:
        raise ValueError("need at least one positive weight")
    point = rng.random() * total
    for key, weight in weights.items():
        point -= weight
        if point < 0:
            return key
    return next(reversed(list(weights)))


def clashes(edit: Edit, chosen: Sequence[Edit]) -> bool:
    """Whether ``edit`` overlaps or touches any chosen edit.

    An insertion occupies the word it is inserted before, so two edits always have an
    untouched word between them.
    """

    def footprint(e: Edit) -> tuple[int, int]:
        return e.start, max(e.end, e.start + 1)

    start, end = footprint(edit)
    return any(
        start <= other_end and other_start <= end
        for other_start, other_end in map(footprint, chosen)
    )


def match_case(original: str, new: str) -> str:
    """Give ``new`` the casing pattern of ``original``: upper, title, or as is."""
    if len(original) > 1 and original.isupper():
        return new.upper()
    if original[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def apply_edits(context: CorruptionContext, edits: Sequence[Edit]) -> Corrupted:
    """Apply non-clashing ``edits`` to the clean paragraph and label the result.

    Replacements take the casing of the word they replace (except Capitalization edits,
    whose point is the casing) and its trailing space. Deleting words hands their final
    trailing space to the word before. A Missing error is labeled on the word after the
    gap, or the word before it at the end.
    """
    ordered = sorted(edits, key=lambda e: e.start)
    for i, edit in enumerate(ordered):
        if clashes(edit, ordered[:i]):
            raise ValueError(f"edit at {edit.start} clashes with an earlier edit")
    words = list(context.words)
    spaces = [token.space_after for token in context.tokens]

    # Right to left, so earlier clean indices stay valid.
    for edit in reversed(ordered):
        new = list(edit.new)
        if edit.op is Operation.REPLACEMENT and edit.kind is not ErrorKind.CAPITALIZATION:
            new = [match_case(edit.orig[0], new[0]), *new[1:]]
        if edit.orig:
            new_spaces = [True] * (len(new) - 1) + [spaces[edit.end - 1]] if new else []
            if not new and edit.start > 0:
                spaces[edit.start - 1] = spaces[edit.end - 1]
        else:
            new_spaces = [True] * len(new)
        words[edit.start : edit.end] = new
        spaces[edit.start : edit.end] = new_spaces

    records: list[EditRecord] = []
    spans: list[WordSpan] = []
    offset = 0
    for edit in ordered:
        start = edit.start + offset
        end = start + len(edit.new)
        offset += len(edit.new) - len(edit.orig)
        records.append(EditRecord(start=start, end=end, kind=edit.kind, op=edit.op, orig=edit.orig))
        if end > start:
            spans.append(WordSpan(start, end, edit.kind))
        else:
            anchor = missing_anchor(start, word_count=len(words))
            spans.append(WordSpan(anchor, anchor + 1, edit.kind))

    labels = spans_to_labels(spans, word_count=len(words))
    return Corrupted(tuple(words), tuple(spaces), tuple(labels), tuple(records))


def choose_edits(
    context: CorruptionContext, registry: CorruptorRegistry, config: GenConfig, rng: Rng
) -> list[Edit]:
    """Sample an error count, then ask weighted corruptors for that many edits.

    A corruptor that finds no site is dropped for this paragraph and another kind is
    tried; an edit that clashes with the chosen ones is discarded. The paragraph may end
    up with fewer errors than sampled.
    """
    count = weighted_choice(config.errors_per_paragraph, rng)
    weights = {kind: w for kind in registry.kinds() if (w := config.weight(kind)) > 0}
    chosen: list[Edit] = []
    attempts = count * ATTEMPTS_PER_ERROR
    while len(chosen) < count and weights and attempts > 0:
        attempts -= 1
        kind = weighted_choice(weights, rng)
        edit = registry.get(kind).corrupt(context, rng)
        if edit is None:
            del weights[kind]
        elif not clashes(edit, chosen):
            chosen.append(edit)
    return chosen


def corrupt_paragraph(
    context: CorruptionContext, registry: CorruptorRegistry, config: GenConfig, rng: Rng
) -> Corrupted:
    """Corrupt one clean paragraph with 0 or more errors drawn from ``config``."""
    return apply_edits(context, choose_edits(context, registry, config, rng))


def revert_edits(words: Sequence[str], edits: Sequence[EditRecord]) -> tuple[str, ...]:
    """Undo ``edits`` on noisy ``words``, giving back the clean words."""
    restored = list(words)
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        restored[edit.start : edit.end] = edit.orig
    return tuple(restored)
