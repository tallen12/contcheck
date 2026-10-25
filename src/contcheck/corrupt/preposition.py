"""Preposition: swap a preposition for a commonly confused one, or drop or add one."""

from collections.abc import Callable, Mapping

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation

# Each preposition maps to the ones writers confuse it with. Subordinating conjunctions
# that share the IN tag (because, although, if, that) are absent, so they are never sites.
CONFUSIONS: Mapping[str, tuple[str, ...]] = {
    "in": ("on", "at", "into"),
    "on": ("in", "at"),
    "at": ("in", "on", "to"),
    "to": ("for", "at", "into"),
    "for": ("to", "of"),
    "of": ("for", "from", "about"),
    "with": ("by", "from"),
    "by": ("with", "from"),
    "from": ("of", "by"),
    "about": ("of", "on"),
    "into": ("in", "to"),
    "onto": ("on", "in"),
}
# Dropping these between a verb and its object leaves the verb with an object it cannot
# take (waited the bus, listened the music). ``in`` and ``on`` are left out: without them
# a time phrase is often still fine (met Monday).
DROPPABLE = frozenset({"to", "at", "about", "with", "from", "for", "of"})
# A noun phrase start a dropped preposition leaves the verb facing. Pronouns are left
# out: gave to him and bought for her become the fine gave him and bought her.
OBJECT_START_TAGS = frozenset({"DT", "PRP$", "NN", "NNS", "NNP", "JJ"})
# Verbs that take a direct object, with the preposition learners wrongly put after them.
UNNEEDED_AFTER: Mapping[str, str] = {
    "discuss": "about",
    "mention": "about",
    "describe": "about",
    "emphasize": "on",
    "lack": "of",
    "resemble": "to",
    "enter": "to",
    "reach": "to",
    "visit": "to",
    "contact": "to",
    "approach": "to",
    "marry": "with",
}
INSERT_BEFORE_TAGS = OBJECT_START_TAGS | {"PRP"}
SWAP_SHARE = 0.7
"""Share of edits that try a swap first; the rest try a drop or an insertion first."""

type Proposal = Callable[[CorruptionContext, Rng], Edit | None]


def _is_preposition(context: CorruptionContext, i: int) -> bool:
    if context.words[i].lower() not in CONFUSIONS:
        return False
    match context.tags[i]:
        case "IN":
            return True
        case "TO":
            # TO also marks infinitives (to go); spaCy tags prepositional to IN, but
            # other taggers use TO for both, so look at what follows.
            return i + 1 < len(context.tokens) and not context.tags[i + 1].startswith("VB")
        case _:
            return False


def swap_preposition(context: CorruptionContext, rng: Rng) -> Edit | None:
    """Replace a preposition by one it is commonly confused with (in for on)."""
    sites = [i for i in range(len(context.tokens)) if _is_preposition(context, i)]
    if not sites:
        return None
    i = rng.choice(sites)
    word = context.words[i]
    new = rng.choice(CONFUSIONS[word.lower()])
    return Edit(i, i + 1, ErrorKind.PREPOSITION, Operation.REPLACEMENT, (word,), (new,))


def drop_preposition(context: CorruptionContext, rng: Rng) -> Edit | None:
    """Delete a preposition between a verb and a noun phrase (waited for the bus)."""
    sites = [
        i
        for i in range(1, len(context.tokens) - 1)
        if _is_preposition(context, i)
        and context.words[i].lower() in DROPPABLE
        and not context.tokens[i].sentence_start
        and context.tags[i - 1].startswith("VB")
        and context.tags[i + 1] in OBJECT_START_TAGS
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    return Edit(i, i + 1, ErrorKind.PREPOSITION, Operation.MISSING, (context.words[i],), ())


def add_preposition(context: CorruptionContext, rng: Rng) -> Edit | None:
    """Insert a preposition between a transitive verb and its object (discuss about it)."""
    sites = [
        i
        for i in range(1, len(context.tokens))
        if context.tags[i - 1].startswith("VB")
        and context.lemmas[i - 1].lower() in UNNEEDED_AFTER
        and context.tags[i] in INSERT_BEFORE_TAGS
        and not context.tokens[i].sentence_start
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    new = UNNEEDED_AFTER[context.lemmas[i - 1].lower()]
    return Edit(i, i, ErrorKind.PREPOSITION, Operation.UNNECESSARY, (), (new,))


class PrepositionCorruptor:
    """Mostly swaps one preposition, sometimes drops or adds one where that is clearly
    wrong, trying the other operations if the first has no site."""

    kind = ErrorKind.PREPOSITION

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        others: list[Proposal] = [drop_preposition, add_preposition]
        if rng.random() < SWAP_SHARE:
            order = [swap_preposition, *others]
        else:
            first = rng.choice(others)
            order = [first, swap_preposition, *(p for p in others if p is not first)]
        for proposal in order:
            edit = proposal(context, rng)
            if edit is not None:
                return edit
        return None
