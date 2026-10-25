"""The tense profile of a document, from the Penn tags of its finite verbs."""

from collections.abc import Iterable

from contcheck.analysis.models import TenseHistogram, Token

PAST_TAGS = frozenset({"VBD"})
PRESENT_TAGS = frozenset({"VBP", "VBZ"})
MODAL_TAGS = frozenset({"MD"})


def tense_histogram(tokens: Iterable[Token]) -> TenseHistogram:
    """Share of finite verbs that are past (VBD), present (VBP, VBZ), modal (MD) or other.

    A document without finite verbs gets all zeros, so no tense counts as dominant.
    """
    past = present = modal = other = 0
    for token in tokens:
        if not token.finite:
            continue
        if token.tag in PAST_TAGS:
            past += 1
        elif token.tag in PRESENT_TAGS:
            present += 1
        elif token.tag in MODAL_TAGS:
            modal += 1
        else:
            other += 1
    total = past + present + modal + other
    if total == 0:
        return TenseHistogram(0.0, 0.0, 0.0, 0.0)
    return TenseHistogram(past / total, present / total, modal / total, other / total)
