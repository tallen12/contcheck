"""Conjunction: swap a conjunction for one that breaks the meaning or the grammar."""

from collections.abc import Mapping

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation

# Coordinators, tagged CC. A sentence-initial And or But is skipped: swapping one for the
# other there is usually still fine.
COORDINATING: Mapping[str, tuple[str, ...]] = {
    "and": ("but", "or"),
    "but": ("and", "so"),
    "or": ("and", "nor"),
    "nor": ("or",),
}
# Subordinators, tagged IN. ``despite`` takes a noun phrase, not a clause, so it is
# always wrong in their place. than/then belongs to RealWord, that and whether are left
# alone because swapping them is rarely a real error.
SUBORDINATING: Mapping[str, tuple[str, ...]] = {
    "because": ("although", "so"),
    "although": ("because", "despite"),
    "though": ("because", "despite"),
    "while": ("because", "despite"),
    "unless": ("if", "until"),
    "if": ("unless",),
    "until": ("unless",),
}
# ``so`` joining clauses is tagged CC, IN or RB depending on the tagger; after a comma
# it is the conjunction (it rained, so we stayed).
SO_SWAPS = ("because", "but")


def _swaps(context: CorruptionContext, i: int) -> tuple[str, ...]:
    word = context.words[i].lower()
    tag = context.tags[i]
    if word == "so":
        return SO_SWAPS if i > 0 and context.words[i - 1] == "," else ()
    if tag == "CC" and not context.tokens[i].sentence_start:
        return COORDINATING.get(word, ())
    if tag == "IN":
        return SUBORDINATING.get(word, ())
    return ()


class ConjunctionCorruptor:
    """Replaces one conjunction by another that changes the sense or the grammar
    (because for although, but for and)."""

    kind = ErrorKind.CONJUNCTION

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        sites = [(i, swaps) for i in range(len(context.tokens)) if (swaps := _swaps(context, i))]
        if not sites:
            return None
        i, swaps = rng.choice(sites)
        word = context.words[i]
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (rng.choice(swaps),))
