"""Helpers shared by the verb corruptors: finding a verb's subject and forms of *be*."""

from collections.abc import Mapping
from enum import Enum, auto
from typing import assert_never

from contcheck.corrupt.context import CorruptionContext

NOUN_SINGULAR_TAGS = frozenset({"NN", "NNP"})
NOUN_PLURAL_TAGS = frozenset({"NNS", "NNPS"})
THIRD_SINGULAR_PRONOUNS = frozenset({"he", "she", "it"})
OTHER_PRONOUNS = frozenset({"you", "we", "they"})
# Between a subject and its verb only these may appear, or the guess is unsafe.
SKIPPABLE_TAGS = frozenset({"RB", "MD", "VBZ", "VBP", "VBD", "VB"})
# Tags inside a noun phrase, walked over to find the word before the phrase.
NOUN_PHRASE_TAGS = frozenset(
    {"DT", "PDT", "CD", "PRP$", "POS", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"}
)
# A noun phrase after one of these is an object: in "the list of items was", "items"
# follows "of" and is not the subject.
PREPOSITION_TAGS = frozenset({"IN", "TO"})
# Words that make a following "were" subjunctive ("if I were", "I wish she were"), where
# "were" after a singular subject is correct.
SUBJUNCTIVE_TRIGGERS = frozenset({"if", "wish", "wished", "though", "unless", "suppose"})

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
    pronoun. Anything else in between, or a noun phrase that follows a preposition (the
    object in "the list of items was"), gives ``UNKNOWN``, so callers skip the site rather
    than produce an agreement error they did not intend.
    """
    for i in range(verb - 1, -1, -1):
        token = context.tokens[i]
        if token.tag == "PRP" or token.tag in NOUN_SINGULAR_TAGS | NOUN_PLURAL_TAGS:
            return Subject.UNKNOWN if _follows_preposition(context, i) else _person(context, i)
        if token.tag not in SKIPPABLE_TAGS or token.sentence_start:
            return Subject.UNKNOWN
    return Subject.UNKNOWN


def _person(context: CorruptionContext, i: int) -> Subject:
    token = context.tokens[i]
    if token.tag in NOUN_SINGULAR_TAGS:
        return Subject.THIRD_SINGULAR
    if token.tag in NOUN_PLURAL_TAGS:
        return Subject.OTHER
    word = token.text.lower()
    if word == "i":
        return Subject.FIRST_SINGULAR
    if word in THIRD_SINGULAR_PRONOUNS:
        return Subject.THIRD_SINGULAR
    if word in OTHER_PRONOUNS:
        return Subject.OTHER
    return Subject.UNKNOWN


def _follows_preposition(context: CorruptionContext, head: int) -> bool:
    """Whether the noun phrase ending at ``head`` comes right after a preposition."""
    if context.tags[head] == "PRP":
        # A pronoun is a phrase on its own; the words before it belong to something else.
        return head > 0 and context.tags[head - 1] in PREPOSITION_TAGS
    i = head
    while i > 0 and not context.tokens[i].sentence_start:
        if context.tags[i - 1] in PREPOSITION_TAGS:
            return True
        if context.tags[i - 1] not in NOUN_PHRASE_TAGS:
            return False
        i -= 1
    return False


def follows_subjunctive_trigger(context: CorruptionContext, verb: int) -> bool:
    """Whether a word like *if* or *wish* comes earlier in the verb's sentence."""
    for i in range(verb - 1, -1, -1):
        if context.words[i].lower() in SUBJUNCTIVE_TRIGGERS:
            return True
        if context.tokens[i].sentence_start:
            return False
    return False


def present_tag_for(subject: Subject) -> str | None:
    """The present-tense Penn tag that agrees with ``subject``."""
    match subject:
        case Subject.THIRD_SINGULAR:
            return "VBZ"
        case Subject.FIRST_SINGULAR | Subject.OTHER:
            return "VBP"
        case Subject.UNKNOWN:
            return None
        case unreachable:
            assert_never(unreachable)


def be_past_to_present(word: str, subject: Subject) -> str | None:
    """Present form of past *be* (``was``/``were``) that keeps agreement, if known.

    ``were`` only becomes ``are`` after a plural or *you* subject: after *I* or a singular
    subject it is the subjunctive ("if I were"), and ``are`` would add an agreement error.
    """
    match word.lower(), subject:
        case "were", Subject.OTHER:
            return "are"
        case "was", Subject.FIRST_SINGULAR:
            return "am"
        case "was", Subject.THIRD_SINGULAR:
            return "is"
        case _:
            return None
