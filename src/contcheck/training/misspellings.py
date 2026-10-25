"""A check on real misspellings: put common ones into clean text and count how many are caught.

Synthetic val only contains the typos the corruptor makes, so it cannot show whether the
model generalizes to the misspellings people actually make. This check takes a bundled
list of common misspellings (``occured``, ``seperate``), swaps them in for the correct
word wherever it appears in clean words of the evaluation records, and measures recall.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files

from contcheck.corpus.record import Record
from contcheck.taxonomy.bio import WordSpan
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.taxonomy.labels import OUTSIDE

MISSPELLINGS = "natural_misspellings.txt"
PER_WORD = 20
"""At most this many probes per correct word, so frequent words such as *which* do not
dominate the score."""


@dataclass(frozen=True, slots=True)
class MisspellingProbe:
    """A record's words with one correct word replaced by a common misspelling of it."""

    words: tuple[str, ...]
    site: int
    misspelling: str


def parse_misspellings(lines: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Map each correct word to its misspellings, from ``correct: miss miss`` lines."""
    misspellings: dict[str, tuple[str, ...]] = {}
    for line in lines:
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        correct, sep, rest = content.partition(":")
        variants = tuple(rest.split())
        if not sep or not correct.strip() or not variants:
            raise ValueError(f"expected 'correct: misspelling ...', got {line!r}")
        misspellings[correct.strip().lower()] = variants
    return misspellings


def load_misspellings() -> dict[str, tuple[str, ...]]:
    """The misspelling list bundled with the package."""
    text = (files("contcheck.training") / "data" / MISSPELLINGS).read_text(encoding="utf-8")
    return parse_misspellings(text.splitlines())


def build_probes(
    records: Iterable[Record],
    misspellings: Mapping[str, tuple[str, ...]],
    *,
    per_word: int = PER_WORD,
) -> list[MisspellingProbe]:
    """One probe per occurrence of a listed word, up to ``per_word`` per word.

    Only lowercase occurrences outside any labeled error qualify, with no error on a
    neighbouring word either, so the probe is the only error near its site. A word's
    misspellings are used in turn.
    """
    used: Counter[str] = Counter()
    probes: list[MisspellingProbe] = []
    for record in records:
        for i, word in enumerate(record.words):
            variants = misspellings.get(word)
            if variants is None or used[word] >= per_word:
                continue
            nearby = record.labels[max(0, i - 1) : i + 2]
            if any(label != OUTSIDE for label in nearby):
                continue
            words = list(record.words)
            words[i] = variants[used[word] % len(variants)]
            probes.append(MisspellingProbe(tuple(words), i, words[i]))
            used[word] += 1
    return probes


def probe_recall(
    probes: Sequence[MisspellingProbe], predicted: Sequence[Sequence[WordSpan]]
) -> float:
    """Share of probes whose site is covered by a predicted Spelling span."""
    if not probes:
        return 0.0
    caught = sum(
        any(
            span.kind is ErrorKind.SPELLING and span.start <= probe.site < span.end
            for span in spans
        )
        for probe, spans in zip(probes, predicted, strict=True)
    )
    return caught / len(probes)
