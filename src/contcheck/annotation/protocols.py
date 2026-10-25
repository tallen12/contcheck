"""Interfaces for annotating (noisy, clean) pairs with labeled edits."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol

from errant.edit import Edit
from spacy.tokens import Doc

from contcheck.annotation.models import Annotation


class Annotator(Protocol):
    """Finds and labels the edits that turn noisy words into clean words.

    Words must come from the ``TextAnalyzer`` that splits record text, so the returned
    indices line up with ``Record.words``.
    """

    def annotate(self, noisy: Sequence[str], clean: Sequence[str]) -> Annotation:
        """Label the edits from ``noisy`` to ``clean``, or say which edit has no kind."""
        ...

    def annotate_many(
        self, pairs: Iterable[tuple[Sequence[str], Sequence[str]]]
    ) -> Iterator[Annotation]:
        """Annotate ``(noisy, clean)`` pairs in order; adapters may batch for speed."""
        ...


class EditExtractor(Protocol):
    """ERRANT's annotator as the adapter uses it: typed edits between two parsed docs.

    ``errant.Annotator`` satisfies it; tests pass a fake that returns hand-made edits.
    """

    def annotate(self, orig: Doc, cor: Doc) -> Iterable[Edit]:
        """The typed edits that turn ``orig`` into ``cor``."""
        ...
