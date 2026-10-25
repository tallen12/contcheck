"""AdjectiveAdverb: put an adjective where the slot needs an adverb, or the reverse."""

from collections.abc import Mapping

from contcheck.analysis.protocols import WordClassLexicon
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.sampling import shuffled
from contcheck.taxonomy.kinds import ErrorKind, Operation

IRREGULAR_ADVERBS: Mapping[str, str] = {"good": "well"}
IRREGULAR_ADJECTIVES: Mapping[str, str] = {adverb: adj for adj, adverb in IRREGULAR_ADVERBS.items()}
# Adjectives whose -ly word is not their plain adverb (hard/hardly, short/shortly), or
# that are quantifiers and ordinals rather than manner words (several/severally).
MEANING_SHIFTS = frozenset(
    {"hard", "late", "near", "short", "high", "most", "bare", "scarce", "present", "fair"}
    | {"sure", "just", "even", "large", "ready", "pretty", "hot", "several", "many", "few"}
    | {"much", "other", "own", "whole", "first", "last", "next"}
)
VERB_TAGS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})
NOUN_TAGS = frozenset({"NN", "NNS"})
# After these an adjective is the complement (``looks happy``), so it is no error.
LINKING_LEMMAS = frozenset(
    {"be", "seem", "look", "feel", "become", "appear", "sound", "smell", "taste", "get"}
    | {"remain", "stay", "grow", "turn", "keep", "prove", "go", "come", "make", "find"}
)
PREDICATE_LEMMAS = frozenset({"be", "seem", "become", "appear"})
CLAUSE_END_TAGS = frozenset({".", ",", ":", "CC"})
VOWELS = frozenset("aeiou")


def adverb_candidates(adjective: str) -> tuple[str, ...]:
    """Possible -ly adverbs of ``adjective``, most specific spelling rule first."""
    if adjective.endswith("ly"):
        return ()
    candidates: list[str] = []
    if adjective.endswith("y"):
        candidates.append(adjective[:-1] + "ily")  # happy, happily
    if adjective.endswith("le") and adjective[-3:-2] not in VOWELS:
        candidates.append(adjective[:-1] + "y")  # simple, simply
    if adjective.endswith("ic"):
        candidates.append(adjective + "ally")  # basic, basically
    if adjective.endswith("ll"):
        candidates.append(adjective + "y")  # full, fully
    if adjective.endswith("ue"):
        candidates.append(adjective[:-1] + "ly")  # true, truly
    candidates.append(adjective + "ly")
    return tuple(candidates)


def adjective_candidates(adverb: str) -> tuple[str, ...]:
    """Possible adjectives an -ly ``adverb`` derives from, most specific rule first."""
    if not adverb.endswith("ly"):
        return ()
    candidates: list[str] = []
    if adverb.endswith("ically"):
        candidates.append(adverb[:-4])  # basically, basic
    if adverb.endswith("ily"):
        candidates.append(adverb[:-3] + "y")  # happily, happy
    if adverb.endswith("lly"):
        candidates.append(adverb[:-1])  # fully, full
    if adverb.endswith("uly"):
        candidates.append(adverb[:-2] + "e")  # truly, true
    candidates.append(adverb[:-2])  # quickly, quick
    candidates.append(adverb[:-1] + "e")  # simply, simple
    return tuple(candidates)


class AdjectiveAdverbCorruptor:
    """Swaps an adjective and its adverb where the slot makes the swapped word wrong.

    Adverbs become adjectives when they modify a verb (``ran quickly`` to ``ran quick``,
    ``did well`` to ``did good``, ``carefully opened`` to ``careful opened``) or an
    adjective (``really cold`` to ``real cold``). Adjectives become adverbs before a noun
    (``a careful driver`` to ``a carefully driver``) or as the complement of *be*, *seem*,
    *become* or *appear* that ends a clause (``was quiet.`` to ``was quietly.``).

    Derived words must be in the lexicon, so made-up forms are never produced, and words
    that are both adjective and adverb (``fast``, ``kindly``) are left alone.
    """

    kind = ErrorKind.ADJECTIVE_ADVERB

    def __init__(self, lexicon: WordClassLexicon) -> None:
        self._lexicon = lexicon

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        # Stop at the first site with an edit, rather than building every edit to use one.
        for i in shuffled(range(len(context.tokens)), rng):
            if (edit := self._swap(context, i)) is not None:
                return edit
        return None

    def _swap(self, context: CorruptionContext, i: int) -> Edit | None:
        word = context.words[i]
        if not (word.isalpha() and word.islower()):
            return None
        match context.tags[i]:
            case "RB" if self._modifies_verb_or_adjective(context, i):
                new = self._adjective_of(word)
            case "JJ" if self._needs_adjective(context, i):
                new = self._adverb_of(word, irregular=self._before_noun(context, i))
            case _:
                new = None
        if new is None:
            return None
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (new,))

    @staticmethod
    def _modifies_verb_or_adjective(context: CorruptionContext, i: int) -> bool:
        before = context.tokens[i - 1] if i > 0 and not context.tokens[i].sentence_start else None
        after = context.tokens[i + 1] if i + 1 < len(context.tokens) else None
        after_verb = (
            before is not None and before.tag in VERB_TAGS and before.lemma not in LINKING_LEMMAS
        )
        before_head = after is not None and (after.tag in VERB_TAGS or after.tag == "JJ")
        return after_verb or before_head

    @staticmethod
    def _before_noun(context: CorruptionContext, i: int) -> bool:
        return i + 1 < len(context.tokens) and context.tags[i + 1] in NOUN_TAGS

    def _needs_adjective(self, context: CorruptionContext, i: int) -> bool:
        if self._before_noun(context, i):
            return True
        if i + 1 >= len(context.tokens):
            return False
        verb = i - 1
        while verb >= 0 and context.tags[verb] == "RB":  # was really cold
            verb -= 1
        if verb < 0:
            return False
        before = context.tokens[verb]
        return (
            before.tag in VERB_TAGS
            and before.lemma in PREDICATE_LEMMAS
            and context.tags[i + 1] in CLAUSE_END_TAGS
        )

    def _adjective_of(self, adverb: str) -> str | None:
        if adverb in IRREGULAR_ADJECTIVES:
            return IRREGULAR_ADJECTIVES[adverb]
        for candidate in adjective_candidates(adverb):
            if self._lexicon.is_adjective(candidate):
                # A flat adverb (fast, quick, slow) is also correct as the adverb.
                flat = self._lexicon.is_adverb(candidate)
                return None if flat or candidate in MEANING_SHIFTS else candidate
        return None

    def _adverb_of(self, adjective: str, *, irregular: bool) -> str | None:
        """The adverb of ``adjective``; ``irregular`` allows ``good`` to ``well``, which is
        wrong before a noun but fine as a complement (``I am well``)."""
        if adjective in MEANING_SHIFTS:
            return None
        if adjective in IRREGULAR_ADVERBS:
            return IRREGULAR_ADVERBS[adjective] if irregular else None
        for candidate in adverb_candidates(adjective):
            if self._lexicon.is_adverb(candidate):
                # An -ly adjective (kindly, lively) is no error before a noun.
                return None if self._lexicon.is_adjective(candidate) else candidate
        return None
