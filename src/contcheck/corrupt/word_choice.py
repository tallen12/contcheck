"""RealWord from a masked LM: swap a content word for a real word that fits worse.

Confusion sets only teach the tagger a fixed list of pairs. Here a masked LM proposes
words that are grammatical in the slot (same Penn tag) and plausible enough to be a
real slip, and a contrastive filter keeps only those the model clearly rejects next to
the original. Without the filter, many proposals are acceptable alternatives (``docks``
for ``harbour``) and would become false labels.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from contcheck.analysis.lexicon import Lexicon
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.lm.models import MaskedSlot, SlotPrediction
from contcheck.lm.protocols import MaskedScorer
from contcheck.taxonomy.kinds import ErrorKind, Operation

CONTENT_TAGS = frozenset(
    {"NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"}
)
"""Nouns, verbs, adjectives and adverbs; proper nouns are left alone."""
INDEFINITE_ARTICLES = frozenset({"a", "an"})
VOWELS = frozenset("aeiou")


@dataclass(frozen=True, slots=True)
class WordChoiceSettings:
    """Thresholds of the contrastive filter and the cost of each call."""

    top_k: int = 100
    """How many of the model's most probable words to consider at a site."""
    min_ratio: float = 100.0
    """P(original) / P(candidate) must reach this, so the original is clearly preferred."""
    max_candidate_probability: float = 0.001
    """A candidate more probable than this is likely acceptable, whatever the ratio."""
    min_original_probability: float = 0.2
    """Sites where the model barely expects the original say little about the context."""
    sites_per_call: int = 4
    """Sites scored together in one batch; the first with a surviving candidate wins."""
    context_words: int = 48
    """Words of context on each side of the site that the model sees."""


DEFAULT_SETTINGS = WordChoiceSettings()


class WordChoiceCorruptor:
    """Replaces a noun, verb, adjective or adverb by a same-tag word the context rejects.

    A candidate must be a dictionary word with the original's Penn tag, not a form of
    the original's lemma, not a skipped (function) word, not a repeat of a neighbouring
    word, and must not turn a preceding *a* or *an* wrong, which would be a second,
    unlabeled error. It survives only if the original is ``min_ratio`` times more
    probable and the candidate itself stays below ``max_candidate_probability``.
    A pair in ``held_out`` is never produced, so withheld confusions stay out of training.
    """

    kind = ErrorKind.REAL_WORD

    def __init__(
        self,
        scorer: MaskedScorer,
        lexicon: Lexicon,
        *,
        skip: Collection[str] = frozenset(),
        held_out: Collection[frozenset[str]] = frozenset(),
        settings: WordChoiceSettings = DEFAULT_SETTINGS,
    ) -> None:
        self._scorer = scorer
        self._lexicon = lexicon
        self._skip = frozenset(word.lower() for word in skip)
        self._held_out = frozenset(frozenset(w.lower() for w in pair) for pair in held_out)
        self._settings = settings

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        sites = _sample(self._sites(context), self._settings.sites_per_call, rng)
        if not sites:
            return None
        slots = [self._slot(context, i) for i in sites]
        predictions = self._scorer.predict(slots, top_k=self._settings.top_k)
        for i, slot, prediction in zip(sites, slots, predictions, strict=True):
            candidates = self._candidates(context, i, slot, prediction)
            if candidates:
                new = rng.choice(candidates)
                return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (context.words[i],), (new,))
        return None

    def _sites(self, context: CorruptionContext) -> list[int]:
        return [
            i
            for i, token in enumerate(context.tokens)
            if token.tag in CONTENT_TAGS
            and token.text.isascii()
            and token.text.isalpha()
            and token.text.lower() not in self._skip
            # Capitals inside a sentence mark names and titles, which a tag can miss.
            and (token.sentence_start or token.text.islower())
        ]

    def _slot(self, context: CorruptionContext, i: int) -> MaskedSlot:
        start = max(0, i - self._settings.context_words)
        end = i + self._settings.context_words + 1
        spaces = tuple(token.space_after for token in context.tokens)
        return MaskedSlot(
            words=context.words[start:end],
            spaces=spaces[start:end],
            position=i - start,
            candidates=(context.words[i],),
        )

    def _candidates(
        self, context: CorruptionContext, i: int, slot: MaskedSlot, prediction: SlotPrediction
    ) -> list[str]:
        settings = self._settings
        original = prediction.probability(slot.words[slot.position])
        if original is None or original < settings.min_original_probability:
            return []
        word = context.words[i]
        tag = context.tags[i]
        own_lemmas = {context.lemmas[i].lower()} | self._lexicon.lemmas(word.lower(), tag)
        neighbours = {w.lower() for w in context.words[max(0, i - 1) : i + 2]}
        after_article = i > 0 and context.words[i - 1].lower() in INDEFINITE_ARTICLES
        candidates: list[str] = []
        for score in prediction.top:
            # At a sentence start every word is capitalized; elsewhere a capital is a name.
            if not (context.tokens[i].sentence_start or score.word.islower()):
                continue
            new = score.word.lower()
            if (
                new in neighbours
                or new in self._skip
                or new in candidates
                or frozenset({word.lower(), new}) in self._held_out
                or score.probability > settings.max_candidate_probability
                or original < settings.min_ratio * score.probability
                or (after_article and (new[0] in VOWELS) != (word[0].lower() in VOWELS))
            ):
                continue
            lemmas = self._lexicon.lemmas(new, tag)
            if lemmas and not lemmas & own_lemmas:
                candidates.append(new)
        return candidates


def _sample(items: Sequence[int], count: int, rng: Rng) -> list[int]:
    """Up to ``count`` items in random order, by a partial Fisher-Yates shuffle."""
    pool = list(items)
    for j in range(min(count, len(pool))):
        k = j + rng.randrange(len(pool) - j)
        pool[j], pool[k] = pool[k], pool[j]
    return pool[:count]
