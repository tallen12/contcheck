"""Fakes shared across domains, and fixtures that build them.

Each fake implements the same protocol as its real adapter, with small in-memory tables
instead of models, so unit tests stay fast, offline and deterministic.
"""

import random
import re
from collections.abc import Iterable, Iterator, Mapping

import pytest

from contcheck.analysis.models import AnalyzedText, Token
from contcheck.spelling.models import Suggestion

FINITE_TAGS = frozenset({"VBD", "VBP", "VBZ", "MD"})
TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

# Penn tag and lemma per lowercase word; other words are tagged NN with themselves as lemma.
DEFAULT_LEXICON: Mapping[str, tuple[str, str]] = {
    "i": ("PRP", "I"),
    "you": ("PRP", "you"),
    "he": ("PRP", "he"),
    "she": ("PRP", "she"),
    "it": ("PRP", "it"),
    "we": ("PRP", "we"),
    "they": ("PRP", "they"),
    "the": ("DT", "the"),
    "a": ("DT", "a"),
    "an": ("DT", "an"),
    "to": ("TO", "to"),
    "in": ("IN", "in"),
    "on": ("IN", "on"),
    "and": ("CC", "and"),
    "there": ("EX", "there"),
    "their": ("PRP$", "their"),
    "then": ("RB", "then"),
    "than": ("IN", "than"),
    "yesterday": ("NN", "yesterday"),
    "went": ("VBD", "go"),
    "go": ("VBP", "go"),
    "goes": ("VBZ", "go"),
    "walked": ("VBD", "walk"),
    "walk": ("VBP", "walk"),
    "walks": ("VBZ", "walk"),
    "bought": ("VBD", "buy"),
    "was": ("VBD", "be"),
    "is": ("VBZ", "be"),
    "are": ("VBP", "be"),
    "can": ("MD", "can"),
    "apples": ("NNS", "apple"),
    "dogs": ("NNS", "dog"),
    "big": ("JJ", "big"),
    "london": ("NNP", "London"),
}

DEFAULT_INFLECTIONS: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("go", "VBD"): ("went",),
    ("go", "VBP"): ("go",),
    ("go", "VBZ"): ("goes",),
    ("walk", "VBD"): ("walked",),
    ("walk", "VBP"): ("walk",),
    ("walk", "VBZ"): ("walks",),
    ("buy", "VBD"): ("bought",),
    ("buy", "VBP"): ("buy",),
    ("buy", "VBZ"): ("buys",),
    ("be", "VBD"): ("was", "were"),
    ("be", "VBP"): ("are",),
    ("be", "VBZ"): ("is",),
}

DEFAULT_WORDS: Mapping[str, int] = {
    word: 1000 - rank
    for rank, word in enumerate(
        """
        the a an and to in on i you he she it we they there their then than went go goes
        walk walked walks store stores apples apple dog dogs big house yesterday bought buy
        was is are can from form park garden morning quiet quite story night
        """.split()
    )
}


class FakeAnalyzer:
    """A ``TextAnalyzer`` that splits on words and punctuation and tags from a lexicon."""

    def __init__(self, lexicon: Mapping[str, tuple[str, str]] = DEFAULT_LEXICON) -> None:
        self._lexicon = lexicon

    def analyze(self, text: str) -> AnalyzedText:
        matches = list(TOKEN_PATTERN.finditer(text))
        tokens: list[Token] = []
        sentence_start = True
        for i, match in enumerate(matches):
            word = match.group()
            end = match.end()
            next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            if word.isalnum() or "'" in word:
                tag, lemma = self._lexicon.get(word.lower(), ("NN", word.lower()))
            else:
                tag, lemma = ".", word
            tokens.append(
                Token(
                    text=word,
                    tag=tag,
                    lemma=lemma,
                    space_after=next_start > end,
                    sentence_start=sentence_start,
                    finite=tag in FINITE_TAGS,
                )
            )
            sentence_start = word in {".", "!", "?"}
        return AnalyzedText(tuple(tokens))

    def analyze_many(self, texts: Iterable[str]) -> Iterator[AnalyzedText]:
        for text in texts:
            yield self.analyze(text)


class FakeInflector:
    """An ``Inflector`` backed by a ``(lemma, tag) -> forms`` table."""

    def __init__(
        self, table: Mapping[tuple[str, str], tuple[str, ...]] = DEFAULT_INFLECTIONS
    ) -> None:
        self._table = table

    def inflect(self, lemma: str, tag: str) -> tuple[str, ...]:
        return self._table.get((lemma, tag), ())


class FakeDictionary:
    """A ``Dictionary`` over a word-to-frequency table, with plain edit distance."""

    def __init__(self, words: Mapping[str, int] = DEFAULT_WORDS) -> None:
        self._words = words

    def contains(self, word: str) -> bool:
        return word.lower() in self._words

    def suggestions(self, word: str, *, max_distance: int = 2) -> list[Suggestion]:
        found = [
            Suggestion(term, distance, frequency)
            for term, frequency in self._words.items()
            if (distance := edit_distance(word.lower(), term)) <= max_distance
        ]
        return sorted(found, key=lambda s: (s.distance, -s.frequency, s.term))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between ``a`` and ``b``."""
    previous: list[int] = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current: list[int] = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


@pytest.fixture
def analyzer() -> FakeAnalyzer:
    return FakeAnalyzer()


@pytest.fixture
def inflector() -> FakeInflector:
    return FakeInflector()


@pytest.fixture
def dictionary() -> FakeDictionary:
    return FakeDictionary()


@pytest.fixture
def rng() -> random.Random:
    """A seeded ``Rng``; ``random.Random`` implements the protocol."""
    return random.Random(13)
