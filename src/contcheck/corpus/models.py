"""Value types for clean-text sources."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Article(BaseModel):
    """One article from a source corpus, with paragraphs separated by blank lines."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    title: str
    text: str


class RejectReason(StrEnum):
    """Why a paragraph was screened out."""

    NON_WORDS = "non_words"
    MARKUP = "markup"
    LIST = "list"
    ALL_CAPS = "all_caps"
    LONG_SENTENCE = "long_sentence"


@dataclass(frozen=True, slots=True)
class Clean:
    """The paragraph passed every check."""


@dataclass(frozen=True, slots=True)
class Rejected:
    """The paragraph failed a check."""

    reason: RejectReason


type Verdict = Clean | Rejected
