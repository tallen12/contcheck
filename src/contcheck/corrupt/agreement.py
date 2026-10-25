"""SubjectVerbAgreement: give one verb the wrong person or number."""

from contcheck.analysis.protocols import Inflector
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.verbs import BE_PAST_SWAP, BE_PRESENT_SWAP
from contcheck.taxonomy.kinds import ErrorKind, Operation

PRESENT_SWAP = {"VBZ": "VBP", "VBP": "VBZ"}


class AgreementCorruptor:
    """Swaps a finite present verb between third-person singular and the other form
    (``walks`` and ``walk``), keeping its tense; *be* also swaps ``was`` and ``were``."""

    kind = ErrorKind.SUBJECT_VERB_AGREEMENT

    def __init__(self, inflector: Inflector) -> None:
        self._inflector = inflector

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        edits = [
            edit
            for i, token in enumerate(context.tokens)
            if token.finite and token.text.isalpha() and (edit := self._swap(context, i))
        ]
        return rng.choice(edits) if edits else None

    def _swap(self, context: CorruptionContext, i: int) -> Edit | None:
        word = context.words[i]
        tag = context.tags[i]
        if context.lemmas[i] == "be":
            new = BE_PRESENT_SWAP.get(word.lower()) or BE_PAST_SWAP.get(word.lower())
        elif tag in PRESENT_SWAP:
            forms = self._inflector.inflect(context.lemmas[i], PRESENT_SWAP[tag])
            new = forms[0] if forms else None
        else:
            new = None
        if new is None or new.lower() == word.lower():
            return None
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (new,))
