"""PronounForm: give a pronoun the wrong case or form (him for he, me for myself)."""

from collections.abc import Mapping

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation

# Personal pronouns (tagged PRP or PRP$) and their wrong forms. ``you`` and ``it`` have
# one form for both cases. Third-person reflexives are left out: him for himself is still
# grammatical, only about someone else. Contractions are split by the tokenizer, so
# it's and its are never confused here.
FORMS: Mapping[str, tuple[str, ...]] = {
    "i": ("me",),
    "me": ("I", "myself"),
    "my": ("me", "mine"),
    "mine": ("my",),
    "myself": ("me",),
    "he": ("him",),
    "him": ("he",),
    "his": ("him",),
    "she": ("her",),
    "her": ("she",),
    "hers": ("her",),
    "we": ("us",),
    "us": ("we", "ourselves"),
    "our": ("us",),
    "ours": ("our",),
    "ourselves": ("us",),
    "they": ("them",),
    "them": ("they", "their"),
    "their": ("them",),
    "theirs": ("their",),
    "your": ("you",),
    "yours": ("your",),
    "yourself": ("you",),
    "its": ("it",),
}
PERSONAL_TAGS = frozenset({"PRP", "PRP$"})
# After than or as both cases are standard (taller than he, taller than him).
COMPARATIVES = frozenset({"than", "as"})
RELATIVE_AFTER_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS", ","})


def _forms(context: CorruptionContext, i: int) -> tuple[str, ...]:
    word = context.words[i]
    lower = word.lower()
    previous = context.words[i - 1].lower() if i > 0 else ""
    if lower == "who":
        # Only a subject relative (the man who came): whom is right in questions such as
        # who did you see, and who for whom is accepted informally.
        is_relative_subject = (
            i > 0
            and i + 1 < len(context.tokens)
            and context.tags[i - 1] in RELATIVE_AFTER_TAGS
            and context.tokens[i + 1].finite
        )
        return ("whom",) if is_relative_subject else ()
    if context.tags[i] not in PERSONAL_TAGS or previous in COMPARATIVES:
        return ()
    if word == "I" and not context.tokens[i].sentence_start:
        # The applier copies the capital of I, which would give Me inside a sentence.
        return ()
    return FORMS.get(lower, ())


class PronounCorruptor:
    """Replaces one pronoun by another case or form of it, as a single word."""

    kind = ErrorKind.PRONOUN_FORM

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        sites = [(i, forms) for i in range(len(context.tokens)) if (forms := _forms(context, i))]
        if not sites:
            return None
        i, forms = rng.choice(sites)
        word = context.words[i]
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (rng.choice(forms),))
