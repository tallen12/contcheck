"""Value types for LLM annotations of clean paragraphs: the changes a model would make.

``ProposedChange`` and ``ProposedChanges`` validate the model's reply; ``Change`` is a
proposal located in the paragraph; ``AnnotatedParagraph`` is one line of the output.
"""

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from contcheck.taxonomy.kinds import Formality

HASH_LENGTH = 16


def paragraph_hash(text: str) -> str:
    """A stable short id for a paragraph's text (hex SHA-256 prefix)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]


class ProposedChange(BaseModel):
    """One change as the model states it: an exact span of the paragraph and its fix."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    original: str = Field(min_length=1)
    replacement: str
    type: Formality


class ProposedChanges(BaseModel):
    """The model's whole reply; no changes means the paragraph is clean."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    changes: tuple[ProposedChange, ...]


class Change(BaseModel):
    """A proposed change located in the paragraph: ``text[start:end] == original``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    original: str
    replacement: str
    type: Formality


class RejectedChange(BaseModel):
    """A proposed error change that a second model did not confirm, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    change: Change
    reason: str


class AnnotatedParagraph(BaseModel):
    """One output line: a paragraph, the changes a model would make, and the result.

    ``corrected`` is ``text`` with every change applied; it equals ``text`` when
    ``changes`` is empty. After ``corpus verify``, ``changes`` holds only the changes the
    ``verifier`` model confirmed (and informal ones, which it does not check), and
    ``rejected`` the rest. Lines written before verification existed load unverified.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    hash: str = Field(min_length=1)
    text: str
    changes: tuple[Change, ...]
    corrected: str
    model: str
    attempts: int = Field(ge=1)
    verifier: str | None = None
    """The models that checked the error changes, comma-separated in the order they ran;
    ``None`` when the changes were not checked."""
    rejected: tuple[RejectedChange, ...] = ()


class FailureReason(StrEnum):
    """Why a paragraph got no annotation."""

    REQUEST = "request"
    """The server gave no reply after retries."""
    INVALID_JSON = "invalid_json"
    """The reply held no JSON that matches the schema."""
    NOT_FOUND = "not_found"
    """A change's ``original`` is not in the paragraph (or only where another change is)."""
    AMBIGUOUS = "ambiguous"
    """A change's ``original`` occurs more often than changes quote it, so which one is
    meant is unknown."""
    INCOMPLETE = "incomplete"
    """The verdicts do not cover every change exactly once."""


class FailedParagraph(BaseModel):
    """One line of the failure log: a paragraph the model never annotated validly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    hash: str
    reason: FailureReason
    detail: str
    reply: str
    model: str
    attempts: int


@dataclass(frozen=True, slots=True)
class Annotated:
    """The paragraph got a valid annotation."""

    paragraph: AnnotatedParagraph


@dataclass(frozen=True, slots=True)
class Failed:
    """Every attempt failed; the last failure is kept."""

    failure: FailedParagraph


type Outcome = Annotated | Failed
