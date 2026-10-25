"""Attaching formality to decoded errors, and filtering by it."""

from collections.abc import Collection, Sequence
from dataclasses import replace

from contcheck.infer.models import SpanError
from contcheck.infer.protocols import FormalityClassifier
from contcheck.taxonomy.kinds import Formality


def classify_formality(
    words: Sequence[str], errors: Sequence[SpanError], classifier: FormalityClassifier
) -> list[SpanError]:
    """``errors`` with the formality ``classifier`` gives each one."""
    return [replace(error, formality=classifier.classify(words, error)) for error in errors]


def keep_formalities(
    errors: Sequence[SpanError], formalities: Collection[Formality]
) -> list[SpanError]:
    """Only the errors whose formality is in ``formalities``, for example to hide informal
    flags when checking casual writing."""
    return [error for error in errors if error.formality in formalities]
