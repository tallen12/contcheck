"""Helpers shared by the verb corruptors: finding a verb's subject and forms of *be*."""

from collections.abc import Mapping
from enum import Enum, auto

from contcheck.corrupt.context import CorruptionContext

NOUN_SINGULAR_TAGS = frozenset({"NN", "NNP"})
NOUN_PLURAL_TAGS = frozenset({"NNS", "NNPS"})
THIRD_SINGULAR_PRONOUNS = frozenset({"he", "she", "it"})
OTHER_PRONOUNS = frozenset({"you", "we", "they"})
# Between a subject and its verb only these may appear, or the guess is unsafe: in
# "the list of items was", the nearest noun is not the subject.
SKIPPABLE_TAGS = frozenset({"RB", "MD", "VBZ", "VBP", "VBD", "VB"})

BE_PRESENT_TO_PAST: Mapping[str, str] = {"am": "was", "is": "was", "are": "were"}
BE_PAST_SWAP: Mapping[str, str] = {"was": "were", "were": "was"}
BE_PRESENT_SWAP: Mapping[str, str] = {"is": "are", "are": "is", "am": "is"}


class Subject(Enum):
    """Person and number of a verb's subject, as far as agreement needs it."""

    FIRST_SINGULAR = auto()
    THIRD_SINGULAR = auto()
    OTHER = auto()
    UNKNOWN = auto()


def subject_of(context: CorruptionContext, verb: int) -> Subject:
    """Guess the subject of the verb at ``verb`` from the words just before it.

    Looks back within the sentence past adverbs and auxiliaries to the nearest noun or
    pronoun. Anything else in between gives ``UNKNOWN``, so callers skip the site rather
    than produce an agreement error they did not intend.
    """
    for i in range(verb - 1, -1, -1):
        token = context.tokens[i]
        word = token.text.lower()
        if token.tag == "PRP":
            if word == "i":
                return Subject.FIRST_SINGULAR
            if word in THIRD_SINGULAR_PRONOUNS:
                return Subject.THIRD_SINGULAR
            if word in OTHER_PRONOUNS:
                return Subject.OTHER
            return Subject.UNKNOWN
        if token.tag in NOUN_SINGULAR_TAGS:
            return Subject.THIRD_SINGULAR
        if token.tag in NOUN_PLURAL_TAGS:
            return Subject.OTHER
        if token.tag not in SKIPPABLE_TAGS or token.sentence_start:
            return Subject.UNKNOWN
    return Subject.UNKNOWN


def present_tag_for(subject: Subject) -> str | None:
    """The present-tense Penn tag that agrees with ``subject``."""
    match subject:
        case Subject.THIRD_SINGULAR:
            return "VBZ"
        case Subject.FIRST_SINGULAR | Subject.OTHER:
            return "VBP"
        case Subject.UNKNOWN:
            return None


def be_past_to_present(word: str, subject: Subject) -> str | None:
    """Present form of past *be* (``was``/``were``) that keeps agreement, if known."""
    match word.lower(), subject:
        case "were", _:
            return "are"
        case "was", Subject.FIRST_SINGULAR:
            return "am"
        case "was", Subject.THIRD_SINGULAR:
            return "is"
        case _:
            return None
