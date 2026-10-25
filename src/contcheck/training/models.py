"""Evaluation results, as written to the results log."""

from pydantic import BaseModel, ConfigDict, Field, computed_field

from contcheck.taxonomy.kinds import ErrorKind
from contcheck.training.probes import CheckName, ProbeScore

BETA = 0.5
"""F0.5 weights precision twice as much as recall: false alarms annoy more than misses."""


class Scores(BaseModel):
    """Span-level hit counts and the scores derived from them."""

    # The derived scores are written to JSON for readers of the log and recomputed on load.
    model_config = ConfigDict(frozen=True, extra="ignore")

    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)

    @computed_field
    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @computed_field
    @property
    def recall(self) -> float:
        gold = self.true_positives + self.false_negatives
        return self.true_positives / gold if gold else 0.0

    @computed_field
    @property
    def f05(self) -> float:
        p, r = self.precision, self.recall
        if p == 0 and r == 0:
            return 0.0
        return (1 + BETA**2) * p * r / (BETA**2 * p + r)

    def __add__(self, other: Scores) -> Scores:
        return Scores(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )


class EvalResult(BaseModel):
    """Scores of one model or baseline on one evaluation set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    dataset: str
    per_kind: dict[ErrorKind, Scores]
    overall: Scores
    detection: Scores
    """Span boundaries only, kind ignored."""
    confusion: dict[str, dict[str, int]]
    """Gold kind to predicted kind for spans with matching boundaries; ``O`` for none."""
    clean_words: int
    false_alarms_per_1000: float
    """Spans flagged per 1,000 words of paragraphs that have no errors."""
    checks: dict[CheckName, ProbeScore] = Field(default_factory=dict)
    """Probe checks that were run, by name."""
