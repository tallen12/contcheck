"""Confusables: real words that are easily written for each other, weighted for RealWord.

Three sources feed one ``ConfusionMap``:

- hand-written confusion sets (``data/confusion_sets.txt``), kept whole;
- homophones from the CMU Pronouncing Dictionary (``data/homophones.txt``, built by
  ``scripts/build_homophones.py``);
- look-alikes: frequent words within one edit of each other, or two for long words.

Generated pairs skip forms of the same lemma (the inflection corruptors cover those),
derivations that share a stem (inventor and invented), British and American spellings
and words that are not known English words, such as names. Each pair is weighted by its
source and by the frequency of its rarer word, and every word keeps only its strongest
alternatives under a capped total, so common words such as "to" or "the" do not take
most of the edits.
"""

import math
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from itertools import combinations

from contcheck.analysis.protocols import Lemmatizer
from contcheck.spelling.protocols import Dictionary

HOMOPHONES = "homophones.txt"


class ConfusionSource(StrEnum):
    """Where a pair of confusable words comes from, strongest first."""

    HAND_WRITTEN = "hand-written"
    HOMOPHONE = "homophone"
    LOOK_ALIKE = "look-alike"


SOURCE_WEIGHTS: Mapping[ConfusionSource, float] = {
    ConfusionSource.HAND_WRITTEN: 3.0,
    ConfusionSource.HOMOPHONE: 2.0,
    ConfusionSource.LOOK_ALIKE: 1.0,
}
# Two edits apart is a much less likely slip than one.
DISTANCE_TWO_FACTOR = 0.25
# Pairs sharing a prefix this long that differ only near the end are derivations of
# one stem (supported, supporter), which belong to word-form kinds.
STEM_LENGTH = 5
STEM_TAIL = 3
# One word that is another plus this many letters at either end (optional, optionally;
# successful, unsuccessful) is a derivation too.
AFFIX_LENGTH = 2
# British spellings rewritten to American ones, to drop pairs that are spelling variants.
SPELLING_VARIANTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<=\w\w)our(s|ed|ing|ite|ites|able)?$"), r"or\1"),
    (re.compile(r"is(e|ed|es|ing|er|ers|ation|ations)$"), r"iz\1"),
    (re.compile(r"ys(e|ed|es|ing)$"), r"yz\1"),
    (re.compile(r"([bt])re(s?)$"), r"\1er\2"),
    (re.compile(r"ogue(s?)$"), r"og\1"),
    (re.compile(r"ll(ed|ing|er|ers)$"), r"l\1"),
    (re.compile(r"ence(s?)$"), r"ense\1"),
    (re.compile(r"dgement(s?)$"), r"dgment\1"),
    (re.compile(r"^grey"), "gray"),
    (re.compile(r"^tyre"), "tire"),
)


@dataclass(frozen=True, slots=True)
class ConfusionSettings:
    """How look-alikes are found and how much weight each word may carry.

    ``long_word_length`` is the length from which look-alikes may be two edits apart;
    ``None`` keeps them to one edit.
    """

    top_words: int = 25_000
    min_length: int = 3
    long_word_length: int | None = 8
    max_alternatives: int = 8
    word_cap: float = 3.0


DEFAULT_SETTINGS = ConfusionSettings()


@dataclass(frozen=True, slots=True)
class Confusion:
    """Writing ``alternative`` where ``word`` was meant, chosen in proportion to ``weight``."""

    word: str
    alternative: str
    weight: float
    source: ConfusionSource

    @property
    def pair(self) -> frozenset[str]:
        """The two words, without direction, for filtering both directions at once."""
        return frozenset({self.word, self.alternative})


@dataclass(frozen=True, slots=True)
class ConfusionMap:
    """Each lowercase word with its alternatives, heaviest first."""

    by_word: Mapping[str, tuple[Confusion, ...]]

    @classmethod
    def of(cls, confusions: Iterable[Confusion]) -> ConfusionMap:
        """Group confusions by word, in a deterministic order."""
        grouped: dict[str, list[Confusion]] = {}
        for confusion in confusions:
            grouped.setdefault(confusion.word, []).append(confusion)
        return cls(
            {
                word: tuple(sorted(group, key=lambda c: (-c.weight, c.alternative)))
                for word, group in sorted(grouped.items())
            }
        )

    def __len__(self) -> int:
        """The number of directed confusions."""
        return sum(len(group) for group in self.by_word.values())

    def __iter__(self) -> Iterator[Confusion]:
        for group in self.by_word.values():
            yield from group

    def alternatives(self, word: str) -> tuple[Confusion, ...]:
        """What ``word`` (any case) may be confused with; empty if nothing."""
        return self.by_word.get(word.lower(), ())

    def weight(self, word: str) -> float:
        """The total weight of ``word``'s alternatives."""
        return sum(confusion.weight for confusion in self.alternatives(word))

    def pairs(self) -> frozenset[frozenset[str]]:
        """Every confusable pair, without direction."""
        return frozenset(confusion.pair for confusion in self)

    def filter(self, keep: Callable[[Confusion], bool]) -> ConfusionMap:
        """Only the confusions ``keep`` accepts, such as those outside a held-out split."""
        return ConfusionMap.of(confusion for confusion in self if keep(confusion))


@dataclass(frozen=True, slots=True)
class _Candidate:
    source: ConfusionSource
    plausibility: float


def same_stem(a: str, b: str) -> bool:
    """Whether one word extends the other by an affix, or both share a long stem.

    A single added letter (the and they, plan and plane) is a typo, not an affix.
    """
    short, long = sorted((a, b), key=len)
    if len(long) - len(short) >= AFFIX_LENGTH and (long.startswith(short) or long.endswith(short)):
        return True
    prefix = len(_common_prefix(a, b))
    return prefix >= STEM_LENGTH and max(len(a), len(b)) - prefix <= STEM_TAIL


def _common_prefix(a: str, b: str) -> str:
    for i, (char_a, char_b) in enumerate(zip(a, b, strict=False)):
        if char_a != char_b:
            return a[:i]
    return a[: min(len(a), len(b))]


def american_spelling(word: str) -> str:
    """``word`` with common British spellings (colour, realise, centre) made American."""
    for pattern, replacement in SPELLING_VARIANTS:
        word = pattern.sub(replacement, word)
    return word


def distinct_words(a: str, b: str, lemmatizer: Lemmatizer) -> bool:
    """Whether ``a`` and ``b`` are two known words, not forms or spellings of one word."""
    if not (a.isalpha() and b.isalpha()) or a.lower() == b.lower():
        return False
    lemmas_a = lemmatizer.lemmas(a)
    lemmas_b = lemmatizer.lemmas(b)
    return (
        bool(lemmas_a)
        and bool(lemmas_b)
        and not lemmas_a & lemmas_b
        and american_spelling(a) != american_spelling(b)
    )


def most_frequent(frequencies: Mapping[str, int], count: int) -> dict[str, int]:
    """The ``count`` most frequent words, ties broken alphabetically."""
    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:count])


def look_alikes(
    vocabulary: Collection[str],
    *,
    dictionary: Dictionary,
    lemmatizer: Lemmatizer,
    settings: ConfusionSettings,
) -> dict[frozenset[str], float]:
    """Pairs of vocabulary words a typo away from each other, with their plausibility."""
    pairs: dict[frozenset[str], float] = {}
    for word in sorted(vocabulary):
        if len(word) < settings.min_length:
            continue
        long_word = settings.long_word_length is not None and len(word) >= settings.long_word_length
        for suggestion in dictionary.suggestions(word, max_distance=2 if long_word else 1):
            other = suggestion.term
            if (
                other not in vocabulary
                or len(other) < settings.min_length
                or (suggestion.distance > 1 and not long_word_pair(word, other, settings))
                or same_stem(word, other)
                or not distinct_words(word, other, lemmatizer)
            ):
                continue
            plausibility = 1.0 if suggestion.distance <= 1 else DISTANCE_TWO_FACTOR
            pair = frozenset({word, other})
            pairs[pair] = max(pairs.get(pair, 0.0), plausibility)
    return pairs


def long_word_pair(a: str, b: str, settings: ConfusionSettings) -> bool:
    """Whether both words are long enough to be compared two edits apart."""
    limit = settings.long_word_length
    return limit is not None and min(len(a), len(b)) >= limit


def homophone_groups(cmudict_lines: Iterable[str], vocabulary: Collection[str]) -> list[list[str]]:
    """Groups of vocabulary words that the CMU Pronouncing Dictionary says alike.

    Lines are ``word phone phone ...``, with ``word(2)`` for alternative pronunciations and
    ``#`` comments. Only the first pronunciation counts, as the alternatives include
    reduced and regional ones (or as "er", color as "caller"); stress marks count too.
    """
    by_sound: dict[tuple[str, ...], set[str]] = {}
    for line in cmudict_lines:
        fields = line.split("#", 1)[0].split()
        if len(fields) < 2:
            continue
        word = fields[0].lower()
        if word.isalpha() and word in vocabulary:
            by_sound.setdefault(tuple(fields[1:]), set()).add(word)
    groups = {tuple(sorted(words)) for words in by_sound.values() if len(words) > 1}
    return [list(group) for group in sorted(groups)]


def parse_homophone_groups(lines: Iterable[str]) -> list[list[str]]:
    """Groups of words, one per line; ``#`` starts a comment and a word may repeat."""
    groups: list[list[str]] = []
    for line in lines:
        words = line.split("#", 1)[0].lower().split()
        if len(words) > 1:
            groups.append(words)
    return groups


def load_homophone_groups() -> list[list[str]]:
    """The homophone groups bundled with the package."""
    text = (files("contcheck.corrupt") / "data" / HOMOPHONES).read_text(encoding="utf-8")
    return parse_homophone_groups(text.splitlines())


def hand_written_map(confusion_sets: Mapping[str, Sequence[str]]) -> ConfusionMap:
    """The hand-written sets alone, every alternative weighted equally."""
    return ConfusionMap.of(
        Confusion(word, other, 1.0, ConfusionSource.HAND_WRITTEN)
        for word, others in confusion_sets.items()
        for other in others
    )


def build_confusion_map(
    *,
    frequencies: Mapping[str, int],
    dictionary: Dictionary,
    lemmatizer: Lemmatizer,
    confusion_sets: Mapping[str, Sequence[str]],
    homophones: Iterable[Sequence[str]],
    settings: ConfusionSettings = DEFAULT_SETTINGS,
) -> ConfusionMap:
    """Merge hand-written sets, homophones and look-alikes into one weighted map.

    A pair found by several sources keeps its strongest. Its weight is the source weight
    times its plausibility times ``1 + log10`` of how much more frequent its rarer word is
    than the least frequent vocabulary word. Each word keeps its ``max_alternatives``
    heaviest alternatives, scaled down to ``word_cap`` in total.
    """
    top = most_frequent(frequencies, settings.top_words)
    floor = min(top.values(), default=1)
    candidates: dict[frozenset[str], _Candidate] = {}

    def add(pair: frozenset[str], source: ConfusionSource, plausibility: float) -> None:
        new = _Candidate(source, plausibility)
        old = candidates.get(pair)
        if old is None or _strength(new) > _strength(old):
            candidates[pair] = new

    lookalikes = look_alikes(top, dictionary=dictionary, lemmatizer=lemmatizer, settings=settings)
    for pair, plausibility in lookalikes.items():
        add(pair, ConfusionSource.LOOK_ALIKE, plausibility)
    for group in homophones:
        for a, b in combinations(sorted(set(group)), 2):
            if (
                min(len(a), len(b)) > 1
                and a in top
                and b in top
                and distinct_words(a, b, lemmatizer)
            ):
                add(frozenset({a, b}), ConfusionSource.HOMOPHONE, 1.0)
    for word, others in confusion_sets.items():
        for other in others:
            add(frozenset({word, other}), ConfusionSource.HAND_WRITTEN, 1.0)

    def frequency_factor(pair: frozenset[str]) -> float:
        rarer = min(max(frequencies.get(word, floor), floor) for word in pair)
        return 1.0 + math.log10(rarer / floor)

    directed: dict[str, list[Confusion]] = {}
    for pair, candidate in candidates.items():
        weight = _strength(candidate) * frequency_factor(pair)
        a, b = sorted(pair)
        directed.setdefault(a, []).append(Confusion(a, b, weight, candidate.source))
        directed.setdefault(b, []).append(Confusion(b, a, weight, candidate.source))
    return ConfusionMap.of(
        confusion for group in directed.values() for confusion in _cap(group, settings=settings)
    )


def _strength(candidate: _Candidate) -> float:
    return SOURCE_WEIGHTS[candidate.source] * candidate.plausibility


def _cap(group: Sequence[Confusion], *, settings: ConfusionSettings) -> list[Confusion]:
    kept = sorted(group, key=lambda c: (-c.weight, c.alternative))[: settings.max_alternatives]
    total = sum(confusion.weight for confusion in kept)
    scale = min(1.0, settings.word_cap / total) if total > 0 else 1.0
    return [Confusion(c.word, c.alternative, c.weight * scale, c.source) for c in kept]
