"""VerbForm: give one verb the wrong non-finite form for its auxiliary, keeping its lemma."""

from collections.abc import Mapping
from enum import Enum, auto

from contcheck.analysis.protocols import Inflector
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.sampling import shuffled
from contcheck.taxonomy.kinds import ErrorKind, Operation

MAX_LOOKBACK = 3
"""How far back past adverbs (``not``, ``never``, ``already``) to look for the auxiliary."""


AUXILIARY_LEMMAS = frozenset({"be", "have", "do"})


class Governor(Enum):
    """What decides the form of the verb after it."""

    HAVE = auto()
    BE = auto()
    DO = auto()
    MODAL = auto()
    TO = auto()
    NONE = auto()


# Wrong forms to try per (verb tag, governor), in order. Only non-finite slots and a bare
# past verb are touched, so the result is never a plain tense shift (VerbTense) or a
# person/number swap (SubjectVerbAgreement): no VBZ or VBP is ever produced.
WRONG_FORMS: Mapping[tuple[str, Governor], tuple[str, ...]] = {
    ("VBN", Governor.HAVE): ("VBD", "VB"),  # has went, has walk
    ("VBN", Governor.BE): ("VBD", "VB"),  # was took, was walk
    ("VBG", Governor.BE): ("VB",),  # is go
    ("VB", Governor.MODAL): ("VBG", "VBN"),  # can going, will gone
    ("VB", Governor.TO): ("VBG", "VBN"),  # to going, to gone
    ("VB", Governor.DO): ("VBD", "VBG"),  # did went, does going
    ("VBD", Governor.NONE): ("VBN",),  # she seen
}


def governor_of(context: CorruptionContext, verb: int) -> Governor:
    """The auxiliary, modal or infinitive ``to`` just before ``verb``, past adverbs."""
    for i in range(verb - 1, max(verb - 1 - MAX_LOOKBACK, -1), -1):
        token = context.tokens[i]
        if token.tag == "MD":
            return Governor.MODAL
        if token.tag == "TO":
            return Governor.TO
        if token.tag.startswith("VB"):
            match token.lemma:
                case "have":
                    return Governor.HAVE
                case "be":
                    return Governor.BE
                case "do" if token.finite:
                    return Governor.DO
                case _:
                    return Governor.NONE
        if token.tag != "RB" or token.sentence_start:
            return Governor.NONE
    return Governor.NONE


class VerbFormCorruptor:
    """Swaps a verb into a form its auxiliary does not allow: ``has gone`` to ``has went``,
    ``are going`` to ``are go``, ``can swim`` to ``can swimming``, ``to leave`` to
    ``to leaving``, ``did go`` to ``did went``, and a bare past ``saw`` to ``seen``.

    Forms of *be* are never the target, and a form that is also valid in the slot (a
    regular ``walked`` as both past and participle, or British ``has got``) is skipped.
    """

    kind = ErrorKind.VERB_FORM

    def __init__(self, inflector: Inflector) -> None:
        self._inflector = inflector

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        # Stop at the first site with an edit, rather than building every edit to use one.
        for i in shuffled(range(len(context.tokens)), rng):
            if edits := self._edits(context, i):
                return rng.choice(edits)
        return None

    def _edits(self, context: CorruptionContext, i: int) -> list[Edit]:
        token = context.tokens[i]
        if not (token.text.isalpha() and token.text.islower()) or token.lemma == "be":
            return []
        # A bare past is only a form error when it is a main verb: ``did not`` is do-support.
        if token.tag == "VBD" and (not token.finite or token.lemma in AUXILIARY_LEMMAS):
            return []
        wrong_tags = WRONG_FORMS.get((token.tag, governor_of(context, i)), ())
        if not wrong_tags:
            return []
        valid = {form.lower() for form in self._inflector.inflect(token.lemma, token.tag)}
        valid.add(token.text)
        if token.tag == "VBD":
            # ``came`` to ``come`` would read as a correct present after a plural subject.
            valid.update(form.lower() for form in self._inflector.inflect(token.lemma, "VB"))
        edits: list[Edit] = []
        for tag in wrong_tags:
            forms = self._inflector.inflect(token.lemma, tag)
            if forms and forms[0].lower() not in valid:
                new = forms[0].lower()
                valid.add(new)
                edits.append(
                    Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (token.text,), (new,))
                )
        return edits
