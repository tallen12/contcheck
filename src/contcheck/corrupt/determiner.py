"""Determiner: drop, add or swap an article."""

from collections.abc import Callable

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation

ARTICLES = frozenset({"a", "an", "the"})
VOWELS = frozenset("aeiou")
NOUN_PHRASE_START_TAGS = frozenset({"NN", "NNS", "JJ"})
# An article cannot follow these, so one inserted after them is clearly wrong.
INSERT_AFTER_TAGS = frozenset({"IN", "VB", "VBD", "VBP", "VBZ", "VBG", "VBN", "CC", "TO"})

type Proposal = Callable[[CorruptionContext, Rng], Edit | None]


def _is_article(context: CorruptionContext, i: int) -> bool:
    return context.words[i].lower() in ARTICLES and context.tags[i] == "DT"


def _indefinite_for(word: str) -> str:
    return "an" if word[:1].lower() in VOWELS else "a"


def drop_article(context: CorruptionContext, rng: Rng) -> Edit | None:
    """Delete an article inside a sentence, before a word."""
    sites = [
        i
        for i in range(len(context.tokens) - 1)
        if _is_article(context, i)
        and not context.tokens[i].sentence_start
        and context.words[i + 1].isalpha()
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    return Edit(i, i + 1, ErrorKind.DETERMINER, Operation.MISSING, (context.words[i],), ())


def add_article(context: CorruptionContext, rng: Rng) -> Edit | None:
    """Insert an article before a bare noun phrase after a verb, preposition or conjunction.

    Plural nouns get ``a``/``an``, which is always wrong; other phrases get ``the``.
    """
    sites = [
        i
        for i in range(1, len(context.tokens))
        if context.tags[i] in NOUN_PHRASE_START_TAGS
        and context.tags[i - 1] in INSERT_AFTER_TAGS
        and context.words[i].isalpha()
        and context.words[i].islower()
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    word = context.words[i]
    article = _indefinite_for(word) if context.tags[i] == "NNS" else "the"
    return Edit(i, i, ErrorKind.DETERMINER, Operation.UNNECESSARY, (), (article,))


def swap_article(context: CorruptionContext, rng: Rng) -> Edit | None:
    """Replace an article by another: definite for indefinite, or the wrong a/an form."""
    sites = [
        i
        for i in range(len(context.tokens) - 1)
        if _is_article(context, i) and context.words[i + 1][:1].isalpha()
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    word = context.words[i]
    following = context.words[i + 1]
    match word.lower():
        case "the":
            new = _indefinite_for(following)
        case _ if rng.random() < 0.5:
            new = "the"
        case "a":
            new = "an"
        case _:
            new = "a"
    return Edit(i, i + 1, ErrorKind.DETERMINER, Operation.REPLACEMENT, (word,), (new,))


PROPOSALS: tuple[Proposal, ...] = (drop_article, add_article, swap_article)


class DeterminerCorruptor:
    """Drops, adds or swaps one article, trying the other operations if one has no site."""

    kind = ErrorKind.DETERMINER

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        proposals = list(PROPOSALS)
        while proposals:
            proposal = rng.choice(proposals)
            edit = proposal(context, rng)
            if edit is not None:
                return edit
            proposals.remove(proposal)
        return None
