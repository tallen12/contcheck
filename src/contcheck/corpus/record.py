"""The JSONL record format shared by every data source and the trainer."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contcheck.taxonomy.kinds import ErrorKind, Operation
from contcheck.taxonomy.labels import LABEL2ID


class Split(StrEnum):
    """Which part of the data a document belongs to. Assigned per document."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class RecordSource(StrEnum):
    """Where a record's errors came from."""

    RULE = "rule"
    LLM = "llm"


class EditRecord(BaseModel):
    """One error in a record, in noisy word indices.

    Replacing noisy words ``start`` to ``end`` (exclusive) by ``orig`` undoes the error.
    Missing errors have ``start == end``: the clean words go into the gap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    kind: ErrorKind
    op: Operation
    orig: tuple[str, ...]

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"invalid range [{self.start}, {self.end})")
        if (self.op is Operation.MISSING) != (self.start == self.end):
            raise ValueError("exactly the Missing edits have an empty noisy range")
        return self


class Record(BaseModel):
    """One labeled paragraph: clean text, its noisy variant, and word-level labels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    paragraph: int = Field(ge=0)
    variant: int = Field(ge=0)
    split: Split
    source: RecordSource
    clean: str
    noisy: str
    words: tuple[str, ...]
    """The noisy text split into words."""
    labels: tuple[str, ...]
    """One BIO label per noisy word."""
    edits: tuple[EditRecord, ...]

    @model_validator(mode="after")
    def _check_labels(self) -> Self:
        if len(self.labels) != len(self.words):
            raise ValueError(f"{len(self.words)} words but {len(self.labels)} labels")
        unknown = sorted(set(self.labels) - LABEL2ID.keys())
        if unknown:
            raise ValueError(f"unknown labels {unknown}")
        if any(edit.end > len(self.words) for edit in self.edits):
            raise ValueError("an edit runs past the last word")
        return self
