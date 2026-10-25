"""Generation settings, read from ``configs/gen.yaml``."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from contcheck.taxonomy.kinds import ErrorKind

DEFAULT_WEIGHT = "default"


class GenConfig(BaseModel):
    """How many errors to put in each paragraph, and of which kinds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    variants_per_paragraph: int = Field(ge=1)
    errors_per_paragraph: dict[int, float]
    """Relative weight of each error count; 0 leaves the paragraph clean."""
    kind_weights: dict[str, float]
    """Relative weight per kind name; ``default`` applies to kinds not listed."""
    adjacent_share: float = Field(default=0.0, ge=0, le=1)
    """Share of paragraphs with two or more errors whose errors sit on neighbouring words."""
    held_out_confusion_share: float = Field(default=0.0, ge=0, lt=1)
    """Share of confusion pairs withheld from training for the held-out check."""
    word_choice_share: float = Field(default=0.0, ge=0, le=1)
    """Share of RealWord edits asked of the masked-LM word-choice corruptor instead of the
    confusion sets; 0 never loads the masked LM."""
    word_choice_model: str = "distilroberta-base"
    """Hugging Face masked LM (RoBERTa tokenizer) that proposes and judges word choices."""

    @field_validator("errors_per_paragraph")
    @classmethod
    def _check_counts(cls, value: dict[int, float]) -> dict[int, float]:
        if not value or any(count < 0 for count in value):
            raise ValueError("error counts must be non-negative and at least one is needed")
        _check_weights(list(value.values()))
        return value

    @field_validator("kind_weights")
    @classmethod
    def _check_kinds(cls, value: dict[str, float]) -> dict[str, float]:
        known = {kind.value for kind in ErrorKind} | {DEFAULT_WEIGHT}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown kinds {unknown}")
        _check_weights(list(value.values()))
        return value

    def weight(self, kind: ErrorKind) -> float:
        """The relative sampling weight of ``kind``; 0 when neither it nor default is set."""
        return self.kind_weights.get(kind.value, self.kind_weights.get(DEFAULT_WEIGHT, 0.0))


def _check_weights(weights: list[float]) -> None:
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative with a positive total")


def load_gen_config(path: Path) -> GenConfig:
    """Read and validate a generation config file."""
    with path.open(encoding="utf-8") as file:
        data: object = yaml.safe_load(file)
    return GenConfig.model_validate(data)
