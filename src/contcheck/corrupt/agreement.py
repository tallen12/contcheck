"""SubjectVerbAgreement: give one verb the wrong person or number."""

from contcheck.analysis.protocols import Inflector
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.sampling import shuffled
from contcheck.corrupt.verbs import (
    BE_PAST_SWAP,
    BE_PRESENT_SWAP,
    Subject,
    follows_subjunctive_trigger,
    subject_of,
)
from contcheck.taxonomy.kinds import ErrorKind, Operation

PRESENT_SWAP = {"VBZ": "VBP", "VBP": "VBZ"}


class AgreementCorruptor:
    """Swaps a finite present verb between third-person singular and the other form
    (``walks`` and ``walk``), keeping its tense; *be* also swaps ``was`` and ``were``.

    The past *be* swap needs a known subject: ``were`` after *I* or a singular subject
    can be a correct subjunctive ("if I were"), so ``was`` only becomes ``were`` after a
    singular subject outside an *if*/*wish* clause, and ``were`` only becomes ``was``
    after a plural or *you* subject.
    """

    kind = ErrorKind.SUBJECT_VERB_AGREEMENT

    def __init__(self, inflector: Inflector) -> None:
        self._inflector = inflector

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        # Try sites in random order and stop at the first that inflects, rather than
        # inflecting every verb to use one.
        sites = [
            i for i, token in enumerate(context.tokens) if token.finite and token.text.isalpha()
        ]
        for i in shuffled(sites, rng):
            if (edit := self._swap(context, i)) is not None:
                return edit
        return None

    def _swap(self, context: CorruptionContext, i: int) -> Edit | None:
        word = context.words[i]
        tag = context.tags[i]
        if context.lemmas[i] == "be":
            new = BE_PRESENT_SWAP.get(word.lower()) or _past_be_swap(context, i)
        elif tag in PRESENT_SWAP:
            forms = self._inflector.inflect(context.lemmas[i], PRESENT_SWAP[tag])
            new = forms[0] if forms else None
        else:
            new = None
        if new is None or new.lower() == word.lower():
            return None
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (new,))


def _past_be_swap(context: CorruptionContext, i: int) -> str | None:
    new = BE_PAST_SWAP.get(context.words[i].lower())
    subject = subject_of(context, i)
    match new:
        case "were" if subject in {Subject.FIRST_SINGULAR, Subject.THIRD_SINGULAR}:
            return None if follows_subjunctive_trigger(context, i) else new
        case "was" if subject is Subject.OTHER:
            return new
        case _:
            return None
