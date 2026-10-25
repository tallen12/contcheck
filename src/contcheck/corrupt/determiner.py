"""Determiner: drop, add or swap an article.

Every proposal must leave text that is wrong whatever the context. Swapping *a* for
*the*, adding *the*, or dropping *the* before a plural or mass noun usually gives correct
English, which would teach the tagger to flag good articles. So the proposals only:
drop *a*/*an* before a singular noun, put *a*/*an* before a plural, and use the wrong
*a*/*an* form.
"""

from collections.abc import Callable

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation

ARTICLES = frozenset({"a", "an", "the"})
INDEFINITE_ARTICLES = frozenset({"a", "an"})
VOWELS = frozenset("aeiou")
ADJECTIVE_TAGS = frozenset({"JJ", "JJR", "JJS"})
# An article cannot follow these, so one inserted after them is clearly wrong.
INSERT_AFTER_TAGS = frozenset({"IN", "VB", "VBD", "VBP", "VBZ", "VBG", "VBN", "CC", "TO"})
# "a few" and "a little" become "few" and "little", which are correct with another meaning.
KEEP_BEFORE = frozenset({"few", "little"})

type Proposal = Callable[[CorruptionContext, Rng], Edit | None]


def _is_article(context: CorruptionContext, i: int) -> bool:
    return context.words[i].lower() in ARTICLES and context.tags[i] == "DT"


def _indefinite_for(word: str) -> str:
    return "an" if word[:1].lower() in VOWELS else "a"


def _phrase_noun_tag(context: CorruptionContext, i: int) -> str | None:
    """The tag of the noun heading the phrase at ``i``, past one adjective, if any."""
    if i < len(context.tags) and context.tags[i] in ADJECTIVE_TAGS:
        i += 1
    if i < len(context.tags) and context.tags[i] in {"NN", "NNS"}:
        return context.tags[i]
    return None


def drop_article(context: CorruptionContext, rng: Rng) -> Edit | None:
    """Delete *a* or *an* inside a sentence, before a singular noun phrase."""
    sites = [
        i
        for i in range(len(context.tokens) - 1)
        if _is_article(context, i)
        and context.words[i].lower() in INDEFINITE_ARTICLES
        and not context.tokens[i].sentence_start
        and context.words[i + 1].isalpha()
        and context.words[i + 1].lower() not in KEEP_BEFORE
        and _phrase_noun_tag(context, i + 1) == "NN"
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    return Edit(i, i + 1, ErrorKind.DETERMINER, Operation.MISSING, (context.words[i],), ())


def add_article(context: CorruptionContext, rng: Rng) -> Edit | None:
    """Insert *a*/*an* before a bare plural noun phrase after a verb, preposition or
    conjunction."""
    sites = [
        i
        for i in range(1, len(context.tokens))
        if context.tags[i - 1] in INSERT_AFTER_TAGS
        and context.words[i].isalpha()
        and context.words[i].islower()
        and _phrase_noun_tag(context, i) == "NNS"
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    article = _indefinite_for(context.words[i])
    return Edit(i, i, ErrorKind.DETERMINER, Operation.UNNECESSARY, (), (article,))


def swap_article(context: CorruptionContext, rng: Rng) -> Edit | None:
    """Replace an article by a wrong one: the other *a*/*an* form, or *a*/*an* for *the*
    before a plural."""
    sites = [
        i
        for i in range(len(context.tokens) - 1)
        if _is_article(context, i)
        and context.words[i + 1][:1].isalpha()
        and (
            context.words[i].lower() in INDEFINITE_ARTICLES
            or _phrase_noun_tag(context, i + 1) == "NNS"
        )
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    word = context.words[i]
    match word.lower():
        case "the":
            new = _indefinite_for(context.words[i + 1])
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
