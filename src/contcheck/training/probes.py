"""Probe checks: swap one word of clean evaluation text and see whether the tagger flags it.

Synthetic val only holds the errors the corruptors made, so it cannot show whether the
model generalizes. A probe check takes a substitution table (correct word to wrong
replacements), swaps a replacement in for a correct word in the evaluation records, and
counts how often the swapped word is flagged.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from contcheck.corpus.record import Record
from contcheck.taxonomy.bio import WordSpan
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.taxonomy.labels import OUTSIDE

PER_WORD = 20
"""At most this many probes per correct word, so frequent words do not dominate a score."""


class CheckName(StrEnum):
    """The probe checks ``evaluate`` runs."""

    REAL_MISSPELLINGS = "real-misspellings"
    """Common real misspellings, which the SymSpell pre-pass owns."""
    HELD_OUT_CONFUSIONS = "held-out-confusions"
    """Confusion pairs withheld from training: does the model generalize?"""
    SEEN_CONFUSIONS = "seen-confusions"
    """Confusion pairs used in training, as the control for the held-out check."""


@dataclass(frozen=True, slots=True)
class Probe:
    """A record's words with the word at ``site`` replaced by ``replacement``."""

    words: tuple[str, ...]
    site: int
    replacement: str


@dataclass(frozen=True, slots=True)
class ProbeSet:
    """The probes of one check and the kind a correct flag would have."""

    kind: ErrorKind
    probes: tuple[Probe, ...]


class ProbeScore(BaseModel):
    """How often the swapped word was flagged at all, and flagged with the right kind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probes: int = Field(ge=0)
    detected: float = Field(ge=0, le=1)
    correct_kind: float = Field(ge=0, le=1)


def substitution_probes(
    records: Iterable[Record],
    substitutions: Mapping[str, Sequence[str]],
    *,
    per_word: int = PER_WORD,
) -> tuple[Probe, ...]:
    """One probe per occurrence of a word in ``substitutions``, up to ``per_word`` each.

    Only lowercase occurrences outside any labeled error qualify, with no error on a
    neighbouring word either, so the probe is the only error near its site. A word's
    replacements are used in turn.
    """
    used: Counter[str] = Counter()
    probes: list[Probe] = []
    for record in records:
        for i, word in enumerate(record.words):
            replacements = substitutions.get(word)
            if replacements is None or len(replacements) == 0 or used[word] >= per_word:
                continue
            if any(label != OUTSIDE for label in record.labels[max(0, i - 1) : i + 2]):
                continue
            words = list(record.words)
            words[i] = replacements[used[word] % len(replacements)]
            probes.append(Probe(tuple(words), i, words[i]))
            used[word] += 1
    return tuple(probes)


def score_probes(probe_set: ProbeSet, predicted: Sequence[Sequence[WordSpan]]) -> ProbeScore:
    """Share of probes whose site is covered by any predicted span, and by one of the
    check's kind."""
    probes = probe_set.probes
    if not probes:
        return ProbeScore(probes=0, detected=0.0, correct_kind=0.0)
    detected = 0
    correct = 0
    for probe, spans in zip(probes, predicted, strict=True):
        covering = [span for span in spans if span.start <= probe.site < span.end]
        detected += bool(covering)
        correct += any(span.kind is probe_set.kind for span in covering)
    return ProbeScore(
        probes=len(probes), detected=detected / len(probes), correct_kind=correct / len(probes)
    )
