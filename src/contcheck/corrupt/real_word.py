"""RealWord: swap a word for a real word it is often confused with."""

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from importlib.resources import files
from itertools import accumulate

from contcheck.corrupt.confusables import ConfusionMap
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation

CONFUSION_SETS = "confusion_sets.txt"


def parse_confusion_sets(lines: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Map each word to the other words of its set. ``#`` starts a comment.

    Raises ``ValueError`` if a word is in two sets or a set has a single word.
    """
    confusions: dict[str, tuple[str, ...]] = {}
    for line in lines:
        words = line.split("#", 1)[0].lower().split()
        if not words:
            continue
        if len(set(words)) < 2:
            raise ValueError(f"confusion set {words} needs two different words")
        for word in words:
            if word in confusions:
                raise ValueError(f"{word!r} is in two confusion sets")
            confusions[word] = tuple(other for other in words if other != word)
    return confusions


def load_confusion_sets() -> dict[str, tuple[str, ...]]:
    """The confusion sets bundled with the package."""
    text = (files("contcheck.corrupt") / "data" / CONFUSION_SETS).read_text(encoding="utf-8")
    return parse_confusion_sets(text.splitlines())


def weighted_index(weights: Sequence[float], rng: Rng) -> int:
    """An index into ``weights``, drawn in proportion to its weight."""
    totals = list(accumulate(weights))
    return min(bisect_right(totals, rng.random() * totals[-1]), len(totals) - 1)


class RealWordCorruptor:
    """Replaces a word by one it is often confused with, such as then for than.

    A confusable word type in the paragraph is drawn by its total weight, so a word that
    occurs five times is not five times as likely; then one of its occurrences, then an
    alternative by weight. The applier gives the alternative the original's capitalization.
    """

    kind = ErrorKind.REAL_WORD

    def __init__(self, confusions: ConfusionMap) -> None:
        self._confusions = confusions

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        sites: dict[str, list[int]] = {}
        for i, word in enumerate(context.words):
            if self._confusions.alternatives(word):
                sites.setdefault(word.lower(), []).append(i)
        if not sites:
            return None
        types = list(sites)
        chosen = types[weighted_index([self._confusions.weight(t) for t in types], rng)]
        i = rng.choice(sites[chosen])
        word = context.words[i]
        alternatives = self._confusions.alternatives(word)
        new = alternatives[weighted_index([c.weight for c in alternatives], rng)].alternative
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (new,))
