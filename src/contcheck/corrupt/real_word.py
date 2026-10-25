"""RealWord: swap a word for a real word it is often confused with."""

from collections.abc import Iterable, Mapping
from importlib.resources import files

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


class RealWordCorruptor:
    """Replaces a word from a confusion set by another member, such as then for than."""

    kind = ErrorKind.REAL_WORD

    def __init__(self, confusions: Mapping[str, tuple[str, ...]]) -> None:
        self._confusions = confusions

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        sites = [i for i, word in enumerate(context.words) if word.lower() in self._confusions]
        if not sites:
            return None
        i = rng.choice(sites)
        word = context.words[i]
        new = rng.choice(self._confusions[word.lower()])
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (new,))
