"""Conjunction: swap a conjunction for one that breaks the grammar.

Swaps that only change the meaning (and for or, if for unless, because for although)
leave correct English the text alone cannot fault, so they are not made.
"""

from collections.abc import Mapping

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation

# Coordinators, tagged CC. ``nor`` needs a negation before it, so it is only put in for
# ``or`` in a sentence without one. A sentence-initial coordinator is skipped.
COORDINATING: Mapping[str, tuple[str, ...]] = {"or": ("nor",)}
NEGATIONS = frozenset({"not", "n't", "no", "never", "neither", "nothing", "none", "nobody"})
# Subordinators, tagged IN. ``despite`` takes a noun phrase, not a clause, so it is
# always wrong in their place.
SUBORDINATING: Mapping[str, tuple[str, ...]] = {
    "although": ("despite",),
    "though": ("despite",),
    "while": ("despite",),
}


def _negated_before(context: CorruptionContext, i: int) -> bool:
    for j in range(i - 1, -1, -1):
        if context.words[j].lower() in NEGATIONS:
            return True
        if context.tokens[j].sentence_start:
            return False
    return False


def _swaps(context: CorruptionContext, i: int) -> tuple[str, ...]:
    word = context.words[i].lower()
    tag = context.tags[i]
    if tag == "CC" and not context.tokens[i].sentence_start:
        return () if _negated_before(context, i) else COORDINATING.get(word, ())
    if tag == "IN":
        return SUBORDINATING.get(word, ())
    return ()


class ConjunctionCorruptor:
    """Replaces one conjunction by another that breaks the grammar (despite for although,
    nor for or without a negation)."""

    kind = ErrorKind.CONJUNCTION

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        sites = [(i, swaps) for i in range(len(context.tokens)) if (swaps := _swaps(context, i))]
        if not sites:
            return None
        i, swaps = rng.choice(sites)
        word = context.words[i]
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (rng.choice(swaps),))
