"""Span-level scores: a hit is the same words and the same kind."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from contcheck.taxonomy.bio import WordSpan
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.taxonomy.labels import OUTSIDE
from contcheck.training.models import EvalResult, Scores


@dataclass(frozen=True, slots=True)
class ParagraphSpans:
    """Gold and predicted spans of one paragraph, and its length in words."""

    words: int
    gold: tuple[WordSpan, ...]
    predicted: tuple[WordSpan, ...]


def score(paragraphs: Iterable[ParagraphSpans], *, name: str, dataset: str) -> EvalResult:
    """Precision, recall and F0.5 per kind and overall, detection-only scores, a confusion
    matrix, and false alarms per 1,000 words of paragraphs without gold errors."""
    hits: Counter[ErrorKind] = Counter()
    false_positives: Counter[ErrorKind] = Counter()
    false_negatives: Counter[ErrorKind] = Counter()
    detection = Scores()
    confusion: dict[str, Counter[str]] = {}
    clean_words = 0
    clean_alarms = 0

    for paragraph in paragraphs:
        gold = set(paragraph.gold)
        predicted = set(paragraph.predicted)
        for span in gold & predicted:
            hits[span.kind] += 1
        for span in predicted - gold:
            false_positives[span.kind] += 1
        for span in gold - predicted:
            false_negatives[span.kind] += 1

        gold_by_range = {(s.start, s.end): s.kind for s in gold}
        predicted_by_range = {(s.start, s.end): s.kind for s in predicted}
        matched = gold_by_range.keys() & predicted_by_range.keys()
        detection += Scores(
            true_positives=len(matched),
            false_positives=len(predicted_by_range.keys() - matched),
            false_negatives=len(gold_by_range.keys() - matched),
        )
        for key, kind in gold_by_range.items():
            row = confusion.setdefault(kind.value, Counter())
            row[predicted_by_range[key].value if key in predicted_by_range else OUTSIDE] += 1
        for key, kind in predicted_by_range.items():
            if key not in gold_by_range:
                confusion.setdefault(OUTSIDE, Counter())[kind.value] += 1

        if not gold:
            clean_words += paragraph.words
            clean_alarms += len(predicted)

    kinds = [
        kind for kind in ErrorKind if hits[kind] or false_positives[kind] or false_negatives[kind]
    ]
    per_kind = {
        kind: Scores(
            true_positives=hits[kind],
            false_positives=false_positives[kind],
            false_negatives=false_negatives[kind],
        )
        for kind in kinds
    }
    overall = sum(per_kind.values(), Scores())
    return EvalResult(
        name=name,
        dataset=dataset,
        per_kind=per_kind,
        overall=overall,
        detection=detection,
        confusion={gold: dict(row) for gold, row in confusion.items()},
        clean_words=clean_words,
        false_alarms_per_1000=1000 * clean_alarms / clean_words if clean_words else 0.0,
    )
