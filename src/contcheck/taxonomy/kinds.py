"""The error kinds and edit operations every other domain builds on.

This module is the single source of truth for the taxonomy: the model's label map, the
corruptor registry and reports all derive from these enums. The kinds follow ERRANT's
categories so real annotated datasets map onto them.
"""

from enum import StrEnum


class ErrorKind(StrEnum):
    """What is wrong with a span of text. Values are the names used in labels and JSONL."""

    SPELLING = "Spelling"
    REAL_WORD = "RealWord"
    CAPITALIZATION = "Capitalization"
    WORD_BOUNDARY = "WordBoundary"
    APOSTROPHE = "Apostrophe"
    NOUN_NUMBER = "NounNumber"
    SUBJECT_VERB_AGREEMENT = "SubjectVerbAgreement"
    VERB_TENSE = "VerbTense"
    VERB_FORM = "VerbForm"
    ADJECTIVE_ADVERB = "AdjectiveAdverb"
    PRONOUN_FORM = "PronounForm"
    DETERMINER = "Determiner"
    PREPOSITION = "Preposition"
    CONJUNCTION = "Conjunction"
    WORD_ORDER = "WordOrder"
    MISSING_WORD = "MissingWord"
    EXTRA_WORD = "ExtraWord"
    SENTENCE_BOUNDARY = "SentenceBoundary"
    PUNCTUATION = "Punctuation"


class Operation(StrEnum):
    """How the noisy text differs from the clean text, as in ERRANT's M/U/R prefixes.

    Kept as metadata next to the kind rather than baked into the label. A missing or
    unnecessary closed-class word keeps the kind named after its class: a dropped *the*
    is ``Determiner`` with ``MISSING``, as are prepositions, conjunctions and punctuation.
    Open-class words and pronouns, whose kinds name a change of form rather than a class,
    are ``MissingWord`` or ``ExtraWord``.
    """

    MISSING = "M"
    UNNECESSARY = "U"
    REPLACEMENT = "R"
