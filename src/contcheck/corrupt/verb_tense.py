"""VerbTense: put one verb in the wrong tense for the document."""

from contcheck.analysis.protocols import Inflector
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.verbs import (
    BE_PRESENT_TO_PAST,
    be_past_to_present,
    present_tag_for,
    subject_of,
)
from contcheck.taxonomy.kinds import ErrorKind, Operation

DOMINANT_SHARE = 0.8
"""A tense error is only detectable where the document mostly uses one tense."""


class VerbTenseCorruptor:
    """Shifts a finite verb from the document's dominant tense to the other one.

    Past documents get a present verb that still agrees with its subject (``went`` to
    ``goes`` after *she*), so the only error is the tense. Present documents get a past
    verb. Documents without a dominant past or present tense are left alone.
    """

    kind = ErrorKind.VERB_TENSE

    def __init__(self, inflector: Inflector) -> None:
        self._inflector = inflector

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        if context.tense.past >= DOMINANT_SHARE:
            candidates = [self._to_present(context, i) for i in self._sites(context, {"VBD"})]
        elif context.tense.present >= DOMINANT_SHARE:
            candidates = [self._to_past(context, i) for i in self._sites(context, {"VBP", "VBZ"})]
        else:
            return None
        edits = [edit for edit in candidates if edit is not None]
        return rng.choice(edits) if edits else None

    @staticmethod
    def _sites(context: CorruptionContext, tags: set[str]) -> list[int]:
        return [
            i
            for i, token in enumerate(context.tokens)
            if token.finite and token.tag in tags and token.text.isalpha()
        ]

    def _to_present(self, context: CorruptionContext, i: int) -> Edit | None:
        word = context.words[i]
        subject = subject_of(context, i)
        if context.lemmas[i] == "be":
            new = be_past_to_present(word, subject)
        else:
            tag = present_tag_for(subject)
            new = None if tag is None else self._first_form(context.lemmas[i], tag)
        return self._edit(i, word, new)

    def _to_past(self, context: CorruptionContext, i: int) -> Edit | None:
        word = context.words[i]
        if context.lemmas[i] == "be":
            new = BE_PRESENT_TO_PAST.get(word.lower())
        else:
            new = self._first_form(context.lemmas[i], "VBD")
        return self._edit(i, word, new)

    def _first_form(self, lemma: str, tag: str) -> str | None:
        forms = self._inflector.inflect(lemma, tag)
        return forms[0] if forms else None

    def _edit(self, i: int, word: str, new: str | None) -> Edit | None:
        if new is None or new.lower() == word.lower():
            return None
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (new,))
